package queue

import "golang.org/x/sys/cpu"

type faq interface {
	Enqueue(index uint64)
	Dequeue() (index uint64)
	InitFull()
	InitEmpty()
}

type LfQueue struct {
	fq               faq
	xxx_cachelinePad cpu.CacheLinePad
	aq               faq
	data             [uint64(1) << order]interface{}
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

func (q *LfQueue) Enqueue(val interface{}) bool {
	index := q.fq.Dequeue()
	if index == uint64max {
		return false
	}
	q.data[index] = val
	q.aq.Enqueue(index)
	return true
}

func (q *LfQueue) Dequeue() interface{} {
	index := q.aq.Dequeue()
	if index == uint64max {
		return false
	}
	val := q.data[index]
	q.fq.Enqueue(index)
	return val
}
