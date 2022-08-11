package msqueue

import (
	"github.com/bytedance/gopkg/collection/lscq"
	"github.com/bytedance/gopkg/lang/fastrand"
	"sync"
	"testing"
)

type lnode struct {
	value interface{}
	next  *lnode
}

type normalQ struct {
	array                      []interface{}
	head                       *lnode
	p1, p2, p3, p4, p5, p6, p7 int64
	tail                       *lnode
	sync.Mutex
}

func newNormalQ(n int) *normalQ {
	nd := lnode{next: nil}
	return &normalQ{
		array: nil,
		head:  &nd,
		tail:  &nd,
	}
}

func (q *normalQ) enque(value interface{}) {
	q.Lock()
	defer q.Unlock()
	nd := lnode{value: value, next: nil}
	q.tail.next = &nd
	q.tail = &nd
}

func (q *normalQ) deque() (interface{}, bool) {
	q.Lock()
	defer q.Unlock()
	if q.head == q.tail {
		return nil, false
	} else {
		next := q.head.next
		q.head = next
		return *next, true
	}
}

func BenchmarkMSQueueReadWrite(b *testing.B) {
	b.Run("50Enqueue50Dequeue/MSQueue", func(b *testing.B) {
		q := NewMSQueue()
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				if fastrand.Uint32n(2) == 0 {
					q.enque(uint64(fastrand.Uint32()))
				} else {
					q.deque()
				}
			}
		})
	})
}

func BenchmarkLSCQueueReadWrite(b *testing.B) {
	b.Run("50Enqueue50Dequeue/LSCQueue", func(b *testing.B) {
		q := lscq.NewUint64()
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				if fastrand.Uint32n(2) == 0 {
					q.Enqueue(uint64(fastrand.Uint32()))
				} else {
					q.Dequeue()
				}
			}
		})
	})
}

func BenchmarkLockQueueReadWrite(b *testing.B) {
	b.Run("50Enqueue50Dequeue/LockQueue", func(b *testing.B) {
		q := newNormalQ(8)
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				if fastrand.Uint32n(2) == 0 {
					q.enque(uint64(fastrand.Uint32()))
				} else {
					q.deque()
				}
			}
		})
	})
}

func BenchmarkChannelReadWrite(b *testing.B) {
	b.Run("50Enqueue50Dequeue/Channel", func(b *testing.B) {
		c := make(chan interface{}, b.N)
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				if fastrand.Uint32n(2) == 0 {
					c <- uint64(fastrand.Uint32())
				} else {
					if len(c) == 0 {
						continue
					}
					<-c
				}
			}
		})
	})
}
