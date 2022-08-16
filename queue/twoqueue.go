package queue

import (
	"golang.org/x/sys/cpu"
	"unsafe"
)

type faq interface {
	Enqueue(index uint64)
	Dequeue() (index uint64)
	InitFull()
	InitEmpty()
}

type LfQueue struct {
	fq   faq
	_    [unsafe.Sizeof(cpu.CacheLinePad{})]byte
	aq   faq
	data [uint64(1) << order]uint64
}

func NewLfQueue(aq, fq faq) *LfQueue {
	lfq := &LfQueue{
		aq: aq,
		fq: fq,
	}
	lfq.aq.InitEmpty()
	lfq.fq.InitFull()
	return lfq
}

func (q *LfQueue) Enqueue(val uint64) bool {
	index := q.fq.Dequeue()
	if index == uint64max {
		return false
	}
	q.data[index] = val
	q.aq.Enqueue(index)
	return true
}

func (q *LfQueue) Dequeue() (data uint64, ok bool) {
	index := q.aq.Dequeue()
	if index == uint64max {
		return 0, false
	}
	val := q.data[index]
	q.fq.Enqueue(index)
	return val, true
}
