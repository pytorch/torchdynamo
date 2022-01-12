#!/usr/bin/env pytest
import torch
from torch.nn import functional as F

import torchdynamo.testing
import torchdynamo.utils
from torchdynamo import eval_frame
from torchdynamo.convert_frame import convert_frame
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
