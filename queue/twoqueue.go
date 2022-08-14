package queue

type faq interface {
	Enqueue(index uint64)
	Dequeue() (index uint64)
	Init()
}

type LfQueue struct {
	n      uint64
	fq, aq faq
}

func NewLfQueue(n uint64) *LfQueue {
	return &LfQueue{}
}
