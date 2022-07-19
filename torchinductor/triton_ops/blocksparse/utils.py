import sys
import torch
import triton

class BCSR():
    def __init__(self, rowptrs, cols, vals) -> None:
        self.rowptrs = rowptrs
        self.cols = cols
        self.vals = vals


class RaggedFormat:
    def __init__(self, rowptrs, cols, default=0):
        self.rowptrs = rowptrs
        self.cols = cols
        self.default = default

    def from_dense_mask(mask, default=0) -> None:        
        rowptrs, cols = to_ragged_format_simple(mask)
        return RaggedFormat(rowptrs, cols, default)

    def copy(self):
        c = RaggedFormat(self.rowptrs, self.cols, self.default)
        return c

    def to_dense(self, a):
        '''
        Return a dense representation of `a`.
        '''
        B, m, n, BLOCK_M, BLOCK_N = a.shape

        M = m * BLOCK_M
        N = n * BLOCK_N

        res = torch.zeros((B, M, N), dtype=a.dtype, device=a.device)

        for i in range(m):
            col_start = self.cols[2*i]
            col_end = self.cols[2*i+1]
            for j in range(n):
                if j < col_start or j >= col_end:
                    block = torch.empty([B, BLOCK_M, BLOCK_N], dtype=a.dtype, device=a.device)
                    block.fill_(self.default)
                else:
                    block = a[:, i, j, 0: BLOCK_M, 0: BLOCK_N]

                res[
                    :,
                    i * BLOCK_M: (i+1) * BLOCK_M, 
                    j * BLOCK_N: (j+1) * BLOCK_N
                ] = block

        return res
        

def cdiv(x, y):
    return (x + y -1) // y


def to_block_format(a, BLOCK_M: int, BLOCK_N: int):
    M, N = a.shape
    outer_m_dim = cdiv(M, BLOCK_M)
    outer_n_dim = cdiv(N, BLOCK_N)
    inner_m_dim = BLOCK_M
    inner_n_dim = BLOCK_N

    res = torch.empty(
        (outer_m_dim, outer_n_dim, inner_m_dim, inner_n_dim),
        dtype=a.dtype,
        device=a.device,
    )

    # TODO - Implement/check for padding
    for outer_m in range(outer_m_dim):
        for outer_n in range(outer_n_dim):
            res[outer_m, outer_n, 0: inner_m_dim, 0: inner_n_dim] = a[
                outer_m * BLOCK_M: outer_m * BLOCK_M + inner_m_dim, outer_n * BLOCK_N: outer_n * BLOCK_N + inner_n_dim
            ]
    return res


def to_block_format_with_mask(a, BLOCK_M: int, BLOCK_N: int):
    assert a.dim() == 2
    M, N = a.shape
    outer_m_dim = cdiv(M, BLOCK_M)
    outer_n_dim = cdiv(N, BLOCK_N)
    inner_m_dim = BLOCK_M
    inner_n_dim = BLOCK_N

    res = torch.empty(
        (outer_m_dim, outer_n_dim, inner_m_dim, inner_n_dim),
        dtype=a.dtype,
        device=a.device,
    )

    mask = torch.ones([outer_m_dim, outer_n_dim], device=a.device)

    # TODO - Implement/check for padding
    for m in range(outer_m_dim):
        for n in range(outer_n_dim):
            block = a[
                m * BLOCK_M: (m+1) * BLOCK_M, 
                n * BLOCK_N: (n+1) * BLOCK_N
            ]
            res[m, n, 0: BLOCK_M, 0: BLOCK_N] = block
            if torch.count_nonzero(block) == 0:
                mask[m, n] = 0
    return (res, mask)


def to_block_format_with_mask_bmm_one_mask(a, BLOCK_M: int, BLOCK_N: int):
    assert a.dim() == 3
    B, M, N = a.shape
    outer_m_dim = cdiv(M, BLOCK_M)
    outer_n_dim = cdiv(N, BLOCK_N)
    inner_m_dim = BLOCK_M
    inner_n_dim = BLOCK_N

    res = torch.empty(
        (B, outer_m_dim, outer_n_dim, inner_m_dim, inner_n_dim),
        dtype=a.dtype,
        device=a.device,
    )

    mask = torch.ones([outer_m_dim, outer_n_dim], device=a.device)

    # TODO - Implement/check for padding
    for m in range(outer_m_dim):
        for n in range(outer_n_dim):
            block = a[
                :,
                m * BLOCK_M: (m+1) * BLOCK_M, 
                n * BLOCK_N: (n+1) * BLOCK_N
            ]
            res[:, m, n, 0: BLOCK_M, 0: BLOCK_N] = block
            if torch.count_nonzero(block) == 0:
                mask[m, n] = 0
    return (res, mask)


def to_triton_blocksparse_format(a, BLOCK_M: int, BLOCK_N: int):
    # assert a.dim() == 3
    assert BLOCK_M == BLOCK_N
    
    M, N = a.shape[-2], a.shape[-1]
    outer_m_dim = cdiv(M, BLOCK_M)
    outer_n_dim = cdiv(N, BLOCK_N)

    mask = torch.ones([outer_m_dim, outer_n_dim], device=a.device, dtype=torch.bool)
    mask = mask[None, :, :]
    res = triton.testing.sparsify_tensor(a, mask, BLOCK_M)
    return (res, mask)


def from_block_format(a):
    # TODO - Implement/check for padding
    B, outer_m_dim, outer_n_dim, BLOCK_M, BLOCK_N = a.shape

    M = outer_m_dim * BLOCK_M
    N = outer_n_dim * BLOCK_N

    res = torch.zeros((B, M, N), dtype=a.dtype, device=a.device)

    for m in range(outer_m_dim):
        for n in range(outer_n_dim):
            res[
                :,
                m * BLOCK_M: (m+1) * BLOCK_M, 
                n * BLOCK_N: (n+1) * BLOCK_N
            ] = a[
                    :, m, n, 0: BLOCK_M, 0: BLOCK_N
                ]
    return res


def get_lower_triangular_mask(m, n):
    mask = torch.tril(torch.ones([m, n], device='cuda'))
    return mask


def to_csr_ptrs(a, device='cuda'):
    m, n = a.shape
    nnz = 0
    for i in range(m):
        for j in range(n):
            if a[i,j] != 0:
                nnz += 1
    
    rowptrs = torch.zeros(m+1, dtype=torch.int, device=device)
    rowptrs[0] = 0
    cols = torch.zeros(nnz, dtype=torch.int, device=device)
    nnz = 0
    for i in range(m):
        for j in range(n):
            if a[i,j] != 0:
                cols[nnz] = j
                nnz += 1
        rowptrs[i+1] = nnz
    assert nnz == torch.sum(a)
    return (rowptrs, cols)


def to_ragged_format_simple(a, device='cuda'):
    m, n = a.shape

    cols = torch.empty(2*m, dtype=torch.int, device=device)
    for i in range(m):
        start = -1
        end = -1
        for j in range(n):
            if j == 0:
                if a[i,j] != 0:
                    start = j

            if j == n-1:
                if a[i,j] != 0:
                    end = j + 1
                    continue

            if a[i,j] != 0:
                if a[i,j-1] == 0:
                    start = j
            elif a[i,j] == 0:
                if a[i,j-1] != 0:
                    end = j
            
        cols[2*i] = start
        cols[2*i+1] = end

    rowptrs = torch.empty(m+1, dtype=torch.int, device=device)
    for i in range(m+1):
        rowptrs[i] = 1
    
    return (rowptrs, cols)


def gen_random_matrix(M, N, BM, BN, density=0.5, dtype=torch.float16, device='cuda'):
    m = cdiv(M, BM)
    n = cdiv(N, BN)
    mask = torch.zeros([m, n], dtype=torch.int, device=device)
    for i in range(m):
        for j in range(n):
            p = torch.rand(1)
            if p[0] < density:
                mask[i,j] = 1

    nnz = torch.sum(mask)
    data = torch.randn([nnz, BM, BN], dtype=dtype, device=device)
    return (mask, data)


def gen_random_matrix_dense_blocks(M, N, BM, BN, density=0.5, dtype=torch.float16, device='cuda'):
    m = cdiv(M, BM)
    n = cdiv(N, BN)
    mask = torch.ones([m, n], dtype=torch.int, device=device)
    data = torch.randn([m, n, BM, BN], dtype=dtype, device=device)
    for i in range(m):
        for j in range(n):
            p = torch.rand(1)
            if p[0] > density:
                mask[i,j] = 0
                data[i,j] = torch.zeros(BM, BN)

    return (mask, data)


def gen_random_mcsr_matrix(M, N, BM, BN, density=1, dtype=torch.float16, device='cuda'):
    m = cdiv(M, BM)
    n = cdiv(N, BN)
    mask = torch.zeros([m, n], dtype=torch.int, device=device)
    data = torch.randn([m, n, BM, BN], dtype=dtype, device=device)
    for i in range(m):
        for j in range(n):
            p = torch.rand(1)
            if p[0] < density:
                mask[i,j] = 1
            else:
                data[i,j] = torch.zeros(BM, BN)

    nnz = torch.sum(mask)
    rowptrs = torch.zeros(m+1, dtype=torch.int, device=device)
    rowptrs[0] = 0
    cols = torch.zeros(nnz, dtype=torch.int, device=device)
    
    nnz = 0
    for i in range(m):
        for j in range(n):
            if mask[i,j] != 0:
                cols[nnz] = j
                nnz += 1
        rowptrs[i+1] = nnz
    assert nnz == torch.sum(mask)
    return BCSR(rowptrs, cols, data)


def gen_lower_triangular_mcsr_matrix(M, N, BM, BN, dtype=torch.float16, device='cuda'):
    m = cdiv(M, BM)
    n = cdiv(N, BN)
    mask = torch.ones([m, n], dtype=torch.int, device=device)
    data = torch.randn([m, n, BM, BN], dtype=dtype, device=device)
    for i in range(m):
        for j in range(n):
            if j > i:
                data[i,j] = torch.zeros(BM, BN)
                mask[i,j] = 0

    nnz = torch.sum(mask)
    rowptrs = torch.zeros(m+1, dtype=torch.int, device=device)
    rowptrs[0] = 0
    cols = torch.zeros(nnz, dtype=torch.int, device=device)
    
    nnz = 0
    for i in range(m):
        for j in range(n):
            if mask[i,j] != 0:
                cols[nnz] = j
                nnz += 1
        rowptrs[i+1] = nnz
    assert nnz == torch.sum(mask)
    return BCSR(rowptrs, cols, data)


def gen_empty_matrix_dense_blocks(M, N, BM, BN, batch_size=1, dtype=torch.float16, device='cuda'):
    m = cdiv(M, BM)
    n = cdiv(N, BN)
    mask = torch.ones([m, n], dtype=torch.int, device=device)
    data = torch.empty([batch_size, m, n, BM, BN], dtype=dtype, device=device)
    return (mask, data)

