/* SPDX-License-Identifier: Apache-2.0 */
/* No-op mlock shim for hosts with a hard RLIMIT_MEMLOCK cap (ulimit -l).
   DeepSpeed's NVMe offload pin-memory pool calls mlock() for ~1.2 GB/rank;
   with the cap at 64 MB that aborts in deepspeed_pin_tensor.cpp. Interposing
   mlock() as a no-op keeps everything functional (pinned memory simply is not
   page-locked). Load with LD_PRELOAD. */

#define _GNU_SOURCE
#include <sys/mman.h>

int mlock(const void *addr, size_t len) { (void)addr; (void)len; return 0; }
int munlock(const void *addr, size_t len) { (void)addr; (void)len; return 0; }
int mlockall(int flags) { (void)flags; return 0; }
int munlockall(void) { return 0; }
