#!/usr/bin/env pytest
import copy
from collections import namedtuple

import torch
from torch.nn import functional as F

import torchdynamo.testing
import torchdynamo.utils
from torchdynamo import eval_frame
from torchdynamo.convert_frame import convert_frame
from torchdynamo.convert_frame import convert_frame_assert
from torchdynamo.testing import same


def _do_paste_mask(masks, boxes, img_h: int, img_w: int, skip_empty: bool = True):
    # from detectron2 mask_ops.py

    device = masks.device

    if skip_empty and not torch.jit.is_scripting():
        x0_int, y0_int = torch.clamp(boxes.min(dim=0).values.floor()[:2] - 1, min=0).to(
            dtype=torch.int32
        )
        x1_int = torch.clamp(boxes[:, 2].max().ceil() + 1, max=img_w).to(
            dtype=torch.int32
        )
        y1_int = torch.clamp(boxes[:, 3].max().ceil() + 1, max=img_h).to(
            dtype=torch.int32
        )
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    if not torch.jit.is_scripting():
        if not masks.dtype.is_floating_point:
            masks = masks.float()
    img_masks = F.grid_sample(masks, grid.to(masks.dtype), align_corners=False)

    if skip_empty and not torch.jit.is_scripting():
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()


def cat(tensors, dim=0):
    # from detectron2 wrappers.py
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def shapes_to_tensor(x, device=None):
    # from detectron2 wrappers.py
    if torch.jit.is_scripting():
        return torch.as_tensor(x, device=device)
    if torch.jit.is_tracing():
        assert all(
            [isinstance(t, torch.Tensor) for t in x]
        ), "Shape should be tensor during tracing!"
        # as_tensor should not be used in tracing because it records a constant
        ret = torch.stack(x)
        if ret.device != device:  # avoid recording a hard-coded device if not necessary
            ret = ret.to(device=device)
        return ret
    return torch.as_tensor(x, device=device)


class Boxes:
    # from detectron2 poolers.py
    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx4 matrix.  Each row is (x1, y1, x2, y2).
        """
        device = (
            tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        )
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, 4)).to(dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == 4, tensor.size()
        self.tensor = tensor

    def __len__(self) -> int:
        return self.tensor.shape[0]

    @property
    def device(self):
        return self.tensor.device


def convert_boxes_to_pooler_format(box_lists):
    # from detectron2 structures.py
    boxes = torch.cat([x.tensor for x in box_lists], dim=0)
    # __len__ returns Tensor in tracing.
    sizes = shapes_to_tensor([x.__len__() for x in box_lists], device=boxes.device)
    indices = torch.repeat_interleave(
        torch.arange(len(box_lists), dtype=boxes.dtype, device=boxes.device), sizes
    )
    return cat([indices[:, None], boxes], dim=1)


ReformerBackwardOutput = namedtuple(
    "ReformerBackwardOutput",
    ["attn_output", "hidden_states", "grad_attn_output", "grad_hidden_states"],
)
ReformerEncoderOutput = namedtuple(
    "ReformerEncoderOutput",
    ["hidden_states", "all_hidden_states", "all_attentions", "past_buckets_states"],
)


class _ReversibleFunction(torch.autograd.Function):
    # taken from modeling_reformer.py in huggingface
    @staticmethod
    def forward(
        ctx,
        hidden_states,
        layers,
        attention_mask,
        head_mask,
        num_hashes,
        all_hidden_states,
        all_attentions,
        past_buckets_states,
        use_cache,
        orig_sequence_length,
        output_hidden_states,
        output_attentions,
    ):
        all_buckets = ()

        # split duplicated tensor
        hidden_states, attn_output = torch.chunk(hidden_states, 2, dim=-1)

        for layer_id, (layer, layer_head_mask) in enumerate(zip(layers, head_mask)):
            if output_hidden_states is True:
                all_hidden_states.append(hidden_states)

            layer_outputs = layer(
                prev_attn_output=attn_output,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=layer_head_mask,
                num_hashes=num_hashes,
                past_buckets_states=past_buckets_states,
                use_cache=use_cache,
                orig_sequence_length=orig_sequence_length,
                output_attentions=output_attentions,
            )

            attn_output = layer_outputs.attn_output
            hidden_states = layer_outputs.hidden_states
            all_buckets = all_buckets + (layer_outputs.buckets,)

            if output_attentions:
                all_attentions.append(layer_outputs.attention_probs)

        # Add last layer
        if output_hidden_states is True:
            all_hidden_states.append(hidden_states)

        # attach params to ctx for backward
        ctx.save_for_backward(attn_output.detach(), hidden_states.detach())
        ctx.layers = layers
        ctx.all_buckets = all_buckets
        ctx.head_mask = head_mask
        ctx.attention_mask = attention_mask

        # Concatenate 2 RevNet outputs
        return torch.cat([attn_output, hidden_states], dim=-1)

    @staticmethod
    def backward(ctx, grad_hidden_states):
        grad_attn_output, grad_hidden_states = torch.chunk(
            grad_hidden_states, 2, dim=-1
        )

        # retrieve params from ctx for backward
        attn_output, hidden_states = ctx.saved_tensors

        # create tuple
        output = ReformerBackwardOutput(
            attn_output=attn_output,
            hidden_states=hidden_states,
            grad_attn_output=grad_attn_output,
            grad_hidden_states=grad_hidden_states,
        )

        # free memory
        del grad_attn_output, grad_hidden_states, attn_output, hidden_states

        layers = ctx.layers
        all_buckets = ctx.all_buckets
        head_mask = ctx.head_mask
        attention_mask = ctx.attention_mask

        for idx, layer in enumerate(layers[::-1]):
            # pop last buckets from stack
            buckets = all_buckets[-1]
            all_buckets = all_buckets[:-1]

            # backprop
            output = layer.backward_pass(
                next_attn_output=output.attn_output,
                hidden_states=output.hidden_states,
                grad_attn_output=output.grad_attn_output,
                grad_hidden_states=output.grad_hidden_states,
                head_mask=head_mask[len(layers) - idx - 1],
                attention_mask=attention_mask,
                buckets=buckets,
            )

        assert all_buckets == (), "buckets have to be empty after backpropagation"
        grad_hidden_states = torch.cat(
            [output.grad_attn_output, output.grad_hidden_states], dim=-1
        )

        # num of return vars has to match num of forward() args
        # return gradient for hidden_states arg and None for other args
        return (
            grad_hidden_states,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class ReformerEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = 0.5
        self.layer_norm = torch.nn.LayerNorm(512, eps=1.0e-12)
        self.layers = []

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=[None] * 6,
        num_hashes=None,
        use_cache=False,
        orig_sequence_length=64,
        output_hidden_states=False,
        output_attentions=False,
    ):
        # hidden_states and attention lists to be filled if wished
        all_hidden_states = []
        all_attentions = []
        past_buckets_states = [((None), (None)) for i in range(len(self.layers))]

        # concat same tensor for reversible ResNet
        hidden_states = torch.cat([hidden_states, hidden_states], dim=-1)
        hidden_states = _ReversibleFunction.apply(
            hidden_states,
            self.layers,
            attention_mask,
            head_mask,
            num_hashes,
            all_hidden_states,
            all_attentions,
            past_buckets_states,
            use_cache,
            orig_sequence_length,
            output_hidden_states,
            output_attentions,
        )

        # Apply layer norm to concatenated hidden states
        hidden_states = self.layer_norm(hidden_states)

        # Apply dropout
        hidden_states = torch.nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        return ReformerEncoderOutput(
            hidden_states=hidden_states,
            all_hidden_states=all_hidden_states,
            all_attentions=all_attentions,
            past_buckets_states=past_buckets_states,
        )


def longformer_chunk(hidden_states, window_overlap=256):
    """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""

    # non-overlapping chunks of size = 2w
    hidden_states = hidden_states.view(
        hidden_states.size(0),
        hidden_states.size(1) // (window_overlap * 2),
        window_overlap * 2,
        hidden_states.size(2),
    )

    # use `as_strided` to make the chunks overlap with an overlap size = window_overlap
    chunk_size = list(hidden_states.size())
    chunk_size[1] = chunk_size[1] * 2 - 1

    chunk_stride = list(hidden_states.stride())
    chunk_stride[1] = chunk_stride[1] // 2
    return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)


class ReproTests(torchdynamo.testing.TestCase):
    def test_do_paste_mask(self):
        torchdynamo.utils.counters.clear()
        with eval_frame.optimize(convert_frame(torchdynamo.testing.CompileCounter())):
            _do_paste_mask(
                torch.randn(1, 1, 28, 28),
                torch.tensor([[0.0, 1, 2, 4]]) * 1,
                427,
                640,
                True,
            )
            _do_paste_mask(
                torch.randn(1, 1, 28, 28),
                torch.tensor([[0.0, 1, 2, 4]]) * 2,
                427,
                640,
                True,
            )
            _do_paste_mask(
                torch.randn(1, 1, 28, 28),
                torch.tensor([[0.0, 1, 2, 4]]) * 3,
                612,
                612,
                True,
            )
            _do_paste_mask(
                torch.randn(1, 1, 28, 28),
                torch.tensor([[0.0, 1, 2, 4]]) * 4,
                612,
                612,
                True,
            )
            _do_paste_mask(
                torch.randn(1, 1, 28, 28),
                torch.tensor([[0.0, 1, 2, 4]]) * 2,
                427,
                640,
                False,
            )
        self.assertGreaterEqual(torchdynamo.utils.counters["frames"]["ok"], 5)
        self.assertEqual(
            torchdynamo.utils.counters["frames"]["total"],
            torchdynamo.utils.counters["frames"]["ok"],
        )

    def test_convert_boxes_to_pooler_format(self):
        boxes1 = [
            Boxes(torch.arange(0, 8).reshape((2, 4))),
            Boxes(torch.arange(8, 16).reshape((2, 4))),
        ]
        boxes2 = [
            Boxes(torch.arange(16, 20).reshape((1, 4))),
            Boxes(torch.arange(20, 24).reshape((1, 4))),
        ]
        correct1 = convert_boxes_to_pooler_format(boxes1)
        correct2 = convert_boxes_to_pooler_format(boxes2)
        cnt = torchdynamo.testing.CompileCounter()
        with eval_frame.optimize(convert_frame(cnt)):
            self.assertTrue(same(convert_boxes_to_pooler_format(boxes1), correct1))
            self.assertTrue(same(convert_boxes_to_pooler_format(boxes2), correct2))

        self.assertEqual(cnt.frame_count, 4)
        self.assertEqual(cnt.op_count, 10)

    def test_boxes_len(self):
        def fn(boxes):
            return len(boxes) + boxes.__len__() + boxes.tensor

        boxes1 = Boxes(torch.arange(0, 8).reshape((2, 4)))
        cnt = torchdynamo.testing.CompileCounter()
        with eval_frame.optimize(convert_frame_assert(cnt)):
            self.assertTrue(same(fn(boxes1), boxes1.tensor + 4.0))

        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    def test_reformer(self):
        input = torch.randn([1, 64, 256])
        model = ReformerEncoder()
        torch.manual_seed(1337)
        correct = copy.deepcopy(model)(input)
        cnt = torchdynamo.testing.CompileCounter()
        with eval_frame.optimize(convert_frame(cnt)):
            torch.manual_seed(1337)
            self.assertTrue(same(model(input), correct))

        self.assertEqual(cnt.frame_count, 4)
        self.assertEqual(cnt.op_count, 9)

    def test_longformer_chunk(self):
        input1 = torch.randn([1, 4096, 1])
        input2 = torch.randn([12, 4096, 64])
        correct1 = longformer_chunk(input1)
        correct2 = longformer_chunk(input2)
        cnt = torchdynamo.testing.CompileCounter()
        with eval_frame.optimize(convert_frame(cnt)):
            self.assertTrue(same(longformer_chunk(input1), correct1))
            self.assertTrue(same(longformer_chunk(input2), correct2))
            self.assertTrue(same(longformer_chunk(input1), correct1))
            self.assertTrue(same(longformer_chunk(input2), correct2))

        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 4)
