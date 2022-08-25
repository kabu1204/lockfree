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

func compareAndSwapSCQNodePointer(addr *scqNode128, old, new scqNode128) (swapped bool) {
	return compareAndSwapUint128((*uint128)(unsafe.Pointer(addr)), old.flag, old.data, new.flag, new.data)
}

func compareAndSwapUint128(addr *uint128, old1, old2, new1, new2 uint64) (swapped bool)

func CASPUint128(addr *uint128, old1, old2, new1, new2 uint64) (swapped bool)

func CASUint128(addr *uint128, old uint128, new uint128) (swapped bool)

func loadSCQNodePointer(addr unsafe.Pointer) (val scqNode128)

func loadUint128(addr *uint128) (val uint128)

func runtimeEnableWriteBarrier() bool

//go:linkname runtimeatomicwb runtime.atomicwb
//go:noescape
func runtimeatomicwb(ptr *unsafe.Pointer, new unsafe.Pointer)

//go:linkname runtimenoescape runtime.noescape
func runtimenoescape(p unsafe.Pointer) unsafe.Pointer

//go:nosplit
func atomicWriteBarrier(ptr *unsafe.Pointer) {
	// For SCQ dequeue only. (fastpath)
	if runtimeEnableWriteBarrier() {
		runtimeatomicwb(ptr, nil)
	}
}
