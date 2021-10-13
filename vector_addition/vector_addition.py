import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
x_ptr, # pointer to the first input vector 
y_ptr, # Pointer to the second input vector
output_ptr, # Pointer to the output vetor
n_elements, # Size of the vector
**meta, # Optional meta parameters for the kernel
    ):
    BLOCK_SIZE = meta['BLOCK_SIZE'] # how many input each program should process
    # There are multiple 'program's processing different data. We identify which 
    # program.
    pid = tl.program_id(axis=0) # We use a 1D launch grid so axis is 0
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the 
    # programs would each access the elements [0:64, 64:128, 128:196, 196: 256].
    # Note that offsets is a list of pointers.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory against out-of-bounds accesses
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input
    # is not a multiple of the block size
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM
    tl.store(output_ptr + offsets, output, mask=mask)
