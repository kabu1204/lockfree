package queue

import (
	"fmt"
	"github.com/bytedance/gopkg/collection/lscq"
	"github.com/bytedance/gopkg/lang/fastrand"
	"github.com/kabu1204/lockfree"
	"github.com/stretchr/testify/assert"
	"golang.org/x/sys/cpu"
	"sync"
	"sync/atomic"
	"testing"
	"unsafe"
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

func TestAtomicValueCopy(t *testing.T) {
	a := atomic.Value{}
	var b atomic.Value
	a.Store(10)
	b = a
	b.Store(20)
	t.Logf("a=%v b=%v", a, b)
	t.Log(atomicNilPointer)
}

func BenchmarkMSQueueReadWrite(b *testing.B) {
	b.Run("50Enqueue50Dequeue/MSQueue", func(b *testing.B) {
		q := NewMSQueue()
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				if fastrand.Uint32n(2) == 0 {
					q.Enque(uint64(fastrand.Uint32()))
				} else {
					q.Deque()
				}
			}
		})
	})
}

func BenchmarkMSQueuePoolReadWrite(b *testing.B) {
	b.Run("50Enqueue50Dequeue/MSQueuePool", func(b *testing.B) {
		q := NewMSQueue()
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				if fastrand.Uint32n(2) == 0 {
					q.EnquePool(uint64(fastrand.Uint32()))
				} else {
					q.DequePool()
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

func TestNewNCQ(t *testing.T) {
	t.Logf("cache line size: %v", unsafe.Sizeof(cpu.CacheLinePad{}))
	a := lockfree.CacheRemap16B(10)
	b := lockfree.CacheRemap16B(11)
	t.Logf("a=%v b=%v", a, b)
}

func TestNcq_Enqueue(t *testing.T) {
	q := ncq{}
	q.InitEmpty()
	t.Log(q)
	for i := 0; i < 16; i++ {
		q.Enqueue(uint64(i))
	}
	t.Log(q)
	q.Enqueue(16)
	t.Log(q)
}

func TestNcq_Dequeue(t *testing.T) {
	q := ncq{}
	q.InitFull()
	t.Log(q)
	for i := 0; i < 16; i++ {
		assert.Equal(t, uint64(i), q.Dequeue())
	}
	assert.Equal(t, uint64max, q.Dequeue())
	t.Log(q)
}

func PrintBit(x uint64) {
	fmt.Printf("%064b\n", x)
}

func TestBit(t *testing.T) {
	b := uint64(64)
	for a := 0; a < 192; a++ {
		if (uint64(a) % b) != (uint64(a) & (b - 1)) {
			fmt.Println("no")
		}
		want := uint64(a) / b
		got := (uint64(a) & (^(b - 1))) >> 6
		if want != got {
			fmt.Printf("no: %v %v %v\n", a, want, got)
		}
	}
	// scqsize = 1 << order
	// cycle = tail / scqsize
	// OR
	// cycle = (tail & ~(scqsize - 1)) >> order
	b = uint64(1 << 6)
	c := b
	PrintBit(b)
	PrintBit(b - 1)
	PrintBit(^(b - 1))
	PrintBit(c)
	PrintBit(c & (^(b - 1)))
}
