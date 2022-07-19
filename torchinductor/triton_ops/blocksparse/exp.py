import sys
import torch
import triton 
import triton.language as tl
from torchinductor.triton_ops.blocksparse.utils import *


@triton.jit
def _exp_1d_kernel(x_rowptrs, x_cols, x_data, y_data, 
                M: tl.constexpr, N: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr,
                # tile sizes, BM needs to divide TM etc
                TM: tl.constexpr, TN: tl.constexpr, 
                use_dense_data: tl.constexpr
                ):
    m = tl.program_id(0)
    block_size = BM * BN

    ## Format specific: how to get `k` would depend on the format
    col_start = tl.load(x_cols + 2*m)
    col_end = tl.load(x_cols + 2*m+1)

    offsets = 0
    ## If data layout is dense - good for debugging
    if use_dense_data:
        offsets += m * BM * N 
    else:
        # TODO: add indexing for sparse data blocks
        pass
    offsets += tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :] 
    x_offsets = x_data + offsets
    y_offsets = y_data + offsets

    for _ in range(col_start, col_end):
        ## Format specific: how to get `k` would depend on the format
        block = tl.load(x_offsets)
        
        ## Kernel specific
        block = tl.exp(block)
        
        ## Format specific: how to get `k` would depend on the format
        tl.store(y_offsets, block)
        x_offsets += block_size
        y_offsets += block_size


def exp_1d(x_mask: RaggedFormat, x_data):
    '''
    Launch a 1D grid to do the computation (blocking rows only).

    2x slower than the 2D blocking kernel for dense matrices of shape > 4096 x 4096.
    '''
    B, m, n, BM, BN = x_data.shape
    M = m * BM
    N = n * BN
    y_data = torch.empty_like(x_data)
    ## Grid size: not blocking the columns. Tow few thread blocks?
    grid = (m, B)    
    _exp_1d_kernel[grid](
        x_mask.rowptrs, x_mask.cols, x_data, y_data,
        M, N, BM, BN, BM, BN, True
    )
    # Same mask is used for y
    return (x_mask, y_data)



@triton.jit
def _exp_kernel(x_rowptrs, x_cols, x_data, y_data, 
                M: tl.constexpr, N: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr,
                # tile sizes, BM needs to divide TM etc
                TM: tl.constexpr, TN: tl.constexpr, 
                use_dense_data: tl.constexpr
                ):
    m = tl.program_id(0)
    n = tl.program_id(1)
    
    ## Format specific: how to get `k` would depend on the format
    col_start = tl.load(x_cols + 2*m)
    col_end = tl.load(x_cols + 2*m+1)
    ## Skip the computation if not a nonzero
    if (n >= col_end) | (n < col_start):
        return 

    block_size = BM * BN

    offsets = 0
    ## If data layout is dense - good for debugging
    if use_dense_data:
        offsets += m * BM * N 
    else:
        # TODO: add indexing for sparse data blocks
        pass
    offsets += tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :] 
    offsets += n * block_size
    x_offsets = x_data + offsets
    y_offsets = y_data + offsets

    ## Format specific: how to get `k` would depend on the format
    block = tl.load(x_offsets)
    
    ## Kernel specific
    block = tl.exp(block)
    
    ## Format specific: how to get `k` would depend on the format
    tl.store(y_offsets, block)


def exp(x_mask: RaggedFormat, x_data):
    '''
    Launch a 2D grid to do the computation.

    Achieved comparable performance with torch.exp for dense matrices of shape > 4096 x 4096.
    '''
    B, m, n, BM, BN = x_data.shape
    M = m * BM
    N = n * BN
    y_data = torch.empty_like(x_data)
    grid = (m, n, B)    
    _exp_kernel[grid](
        x_mask.rowptrs, x_mask.cols, x_data, y_data,
        M, N, BM, BN, BM, BN, True
    )
    # Same mask is used for y
    y_mask = x_mask.copy()
    y_mask.default = 1
    return (y_mask, y_data)

