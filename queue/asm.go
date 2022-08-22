//go:build amd64 || arm64

package queue

import (
	"golang.org/x/sys/cpu"
	"unsafe"
)

const offsetARM64HasATOMICS = unsafe.Offsetof(cpu.ARM64.HasATOMICS)

//go:linkname atomicOrUint32 github.com/kabu1204/lockfree/queue.atomicOr
//go:nosplit
func atomicOrUint32(addr *uint32, v uint32)

//go:linkname atomicAndUint32 github.com/kabu1204/lockfree/queue.atomicAnd
//go:nosplit
func atomicAndUint32(addr *uint32, v uint32)

//go:nosplit
func atomicOrUint64(addr *uint64, v uint64)

//go:nosplit
func atomicAndUint64(addr *uint64, v uint64)

//go:nosplit
func atomicOr(addr *uint32, v uint32)

//go:nosplit
func atomicAnd(addr *uint32, v uint32)

type uint128 [2]uint64

func compareAndSwapUint128(addr *uint128, old1, old2, new1, new2 uint64) (swapped bool)

//go:linkname runtimenoescape runtime.noescape
func runtimenoescape(p unsafe.Pointer) unsafe.Pointer
