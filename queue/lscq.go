package queue

import (
	"github.com/bytedance/gopkg/lang/syncx"
	"golang.org/x/sys/cpu"
	"sync/atomic"
	"unsafe"
)

var scqPool = syncx.Pool{
	New: func() interface{} {
		return NewScqCas2()
	},
}

type Lscq struct {
	head *ScqCas2
	_    [unsafe.Sizeof(cpu.CacheLinePad{}) - 8]byte
	tail *ScqCas2
}

func NewLSCQ() *Lscq {
	cq := NewScqCas2()
	q := &Lscq{
		head: cq,
		tail: cq,
	}
	return q
}

func (q *Lscq) Dequeue() (uint64, bool) {
	for {
		cq := (*ScqCas2)(atomic.LoadPointer((*unsafe.Pointer)(unsafe.Pointer(&q.head))))
		val, ok := cq.Dequeue()
		if ok {
			return val, ok
		}
		if cq.next == nil {
			return 0, false
		}
		atomic.StoreInt64(&cq.threshold, DWResetThreshold)
		val, ok = cq.Dequeue()
		if ok {
			return val, ok
		}
		if atomic.CompareAndSwapPointer((*unsafe.Pointer)(unsafe.Pointer(&q.head)), unsafe.Pointer(cq), unsafe.Pointer(cq.next)) {
			cq = nil
		}
	}
}

func (q *Lscq) Enqueue(val uint64) {
	for {
		cq := (*ScqCas2)(atomic.LoadPointer((*unsafe.Pointer)(unsafe.Pointer(&q.tail))))
		if cq.next != nil {
			atomic.CompareAndSwapPointer((*unsafe.Pointer)(unsafe.Pointer(&q.tail)), unsafe.Pointer(cq), unsafe.Pointer(cq.next))
			continue
		}
		if cq.Enqueue(val) {
			return
		}
		atomicOrUint64(&cq.tail, uint64(1)<<63)
		newScq := scqPool.Get().(*ScqCas2)
		newScq.Enqueue(val)
		if atomic.CompareAndSwapPointer((*unsafe.Pointer)(unsafe.Pointer(&cq.next)), nil, unsafe.Pointer(newScq)) {
			atomic.CompareAndSwapPointer((*unsafe.Pointer)(unsafe.Pointer(&q.tail)), unsafe.Pointer(cq), unsafe.Pointer(newScq))
			return
		}
		newScq.Dequeue()
		scqPool.Put(newScq)
	}
}
