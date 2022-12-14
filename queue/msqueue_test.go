package queue

import (
	"fmt"
	"github.com/bytedance/gopkg/collection/skipset"
	"github.com/bytedance/gopkg/lang/fastrand"
	"github.com/bytedance/gopkg/util/gopool"
	"github.com/kabu1204/lockfree/queue/lscq"
	"github.com/stretchr/testify/assert"
	"golang.org/x/sys/cpu"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"
)

type lnode struct {
	value interface{}
	next  *lnode
}

type normalQ struct {
	array []interface{}
	head  *lnode
	_     [unsafe.Sizeof(cpu.CacheLinePad{})]byte
	tail  *lnode
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

func BenchmarkSCQCas2ReadWrite(b *testing.B) {
	b.Run("50Enqueue50Dequeue/SCQCas2", func(b *testing.B) {
		q := NewScqCas2()
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

func BenchmarkSCQReadWrite(b *testing.B) {
	b.Run("50Enqueue50Dequeue/SCQ", func(b *testing.B) {
		q := NewLfQueue(NewSCQ(), NewSCQ())
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

func BenchmarkSelfLSCQReadWrite(b *testing.B) {
	b.Run("50Enqueue50Dequeue/SelfLSCQ", func(b *testing.B) {
		q := NewLSCQ()
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

func BenchmarkNCQReadWrite(b *testing.B) {
	b.Run("50Enqueue50Dequeue/NCQ", func(b *testing.B) {
		q := NewLfQueue(NewNCQ(), NewNCQ())
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

func BenchmarkNCQ8BReadWrite(b *testing.B) {
	b.Run("50Enqueue50Dequeue/NCQ8B", func(b *testing.B) {
		q := NewLfQueue(NewNCQ8b(), NewNCQ8b())
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

func TestNewNCQ(t *testing.T) {
	t.Logf("cache line size: %v", unsafe.Sizeof(cpu.CacheLinePad{}))
	a := cacheRemap8B(10)
	b := cacheRemap8B(11)
	t.Logf("a=%v b=%v", a, b)
}

func TestNcq_Enqueue(t *testing.T) {
	q := ncq{}
	q.InitEmpty()
	for i := 0; i < int(qsize); i++ {
		q.Enqueue(uint64(i))
	}
	q.Enqueue(16)
}

func TestNcq_Dequeue(t *testing.T) {
	q := ncq{}
	q.InitFull()
	for i := 0; i < int(qsize); i++ {
		assert.Equal(t, uint64(i), q.Dequeue())
	}
	assert.Equal(t, uint64max, q.Dequeue())
}

func TestNcq8b_Enqueue(t *testing.T) {
	q := ncq8b{}
	q.InitEmpty()
	for i := 0; i < int(qsize); i++ {
		q.Enqueue(uint64(i))
	}
	q.Enqueue(16)
}

func TestNcq8b_Dequeue(t *testing.T) {
	q := ncq8b{}
	q.InitFull()
	for i := 0; i < int(qsize); i++ {
		assert.Equal(t, uint64(i), q.Dequeue())
	}
	assert.Equal(t, uint64max, q.Dequeue())
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

type Student struct {
	name string
}

func TestReflect(tt *testing.T) {
	var a int = 50
	v := reflect.ValueOf(a) // ??????Value?????????????????????50
	t := reflect.TypeOf(a)  // ??????Type?????????????????????int
	fmt.Println(v, t, v.Type(), t.Kind())

	var b [5]int = [5]int{5, 6, 7, 8}
	fmt.Println(reflect.TypeOf(b), reflect.TypeOf(b).Kind(), reflect.TypeOf(b).Elem()) // [5]int array int

	var Pupil Student
	p := reflect.ValueOf(Pupil) // ??????ValueOf()?????????????????????Value??????

	fmt.Println(p.Type()) // ??????:Student
	fmt.Println(p.Kind()) // ??????:struct
}

func TestFAA(tt *testing.T) {
	tail := uint64(0)
	t := atomic.AddUint64(&tail, 1)
	tt.Log(tail, t)
}

func TestScq_Enqueue(t *testing.T) {
	t.Log(unused)
	q := NewSCQ()
	q.InitFull()
	//for i := uint64(0); i < qsize; i++ {
	//	q.Enqueue(i)
	//}
	//pp.Println(q)
	//q.Enqueue(6)
	//for _, ent := range q.entries {
	//	PrintEntry(ent)
	//}
	//for i := uint64(0); i < qsize; i++ {
	//	data, ok := q.Dequeue()
	//	assert.True(t, ok)
	//	assert.Equal(t, i, data)
	//}
}

func TestScq_Dequeue(t *testing.T) {
	q := NewLfQueue(NewSCQ(), NewSCQ())
	poolEnqueue := gopool.NewPool("pool", 60, &gopool.Config{})
	poolDequeue := gopool.NewPool("depool", 60, &gopool.Config{})
	m1 := skipset.NewUint64()
	m2 := skipset.NewUint64()
	var wg sync.WaitGroup
	wg.Add(120)
	for i := 0; i < 60; i++ {
		poolEnqueue.Go(func() {
			defer wg.Done()
			for j := 0; j < 1000; j++ {
				val := fastrand.Uint64()
				m1.Add(val)
				q.Enqueue(val)
			}
		})
	}
	for i := 0; i < 60; i++ {
		poolDequeue.Go(func() {
			defer wg.Done()
			for j := 0; j < 1000; j++ {
				for {
					data, ok := q.Dequeue()
					if ok {
						m2.Add(data)
						break
					}
				}
			}
		})
	}
	wg.Wait()
	time.Sleep(1 * time.Second)
	t.Log(poolEnqueue.WorkerCount(), poolDequeue.WorkerCount(), m1.Len(), m2.Len())
	assert.Equal(t, m1.Len(), m2.Len())
}

func TestScqCas2_Dequeue(t *testing.T) {
	//q := NewScqCas2()
	q := lscq.NewUint64()
	poolEnqueue := gopool.NewPool("pool", 600, &gopool.Config{})
	poolDequeue := gopool.NewPool("depool", 600, &gopool.Config{})
	m1 := skipset.NewUint64()
	m2 := skipset.NewUint64()
	var wg sync.WaitGroup
	wg.Add(1200)
	start := time.Now()
	for i := 0; i < 600; i++ {
		poolEnqueue.Go(func() {
			defer wg.Done()
			for j := 0; j < 20000; j++ {
				val := fastrand.Uint64()
				m1.Add(val)
				q.Enqueue(val)
			}
		})
	}
	//wg.Wait()
	//wg.Add(60)
	for i := 0; i < 600; i++ {
		poolDequeue.Go(func() {
			defer wg.Done()
			for j := 0; j < 20000; j++ {
				for {
					data, ok := q.Dequeue()
					if ok {
						m2.Add(data)
						break
					}
				}
			}
		})
	}
	wg.Wait()
	t.Logf("time elapsed: %f ns/op", float64(time.Since(start).Nanoseconds())/float64(2*600*20000))
	t.Log(poolEnqueue.WorkerCount(), poolDequeue.WorkerCount(), m1.Len(), m2.Len())
	assert.Equal(t, m1.Len(), m2.Len())
}

func TestAtomicOR(t *testing.T) {
	a := uint32(0b10101101)
	b := uint32(0b11001011)
	want1 := a | b
	atomicOrUint32(&a, b)
	assert.Equal(t, want1, a)

	c := uint64(0b1011100110110001101011010010110110101001101011010010110100101100)
	d := uint64(0b0101101110101011001011011010111100101010000111001110101010111001)
	want2 := c | d
	atomicOrUint64(&c, d)
	assert.Equal(t, want2, c)

	e := uint64(flagUnsafe | unused)
	f := uint64(2<<(order+2) | flagSafe | 16)
	atomicOrUint64(&f, e)
	assert.Equal(t, uint64(2), (f & ^mask)>>(order+2))
	assert.Equal(t, flagSafe, f&flagSafe)
	assert.Equal(t, unused, f&unused)

	t.Log(cpu.X86)
}

type node128 struct {
	data1 uint64
	data2 uint64
}

func TestCasUint128(t *testing.T) {
	addr := &node128{data1: 123, data2: 456}
	ok := compareAndSwapUint128((*uint128)(unsafe.Pointer(addr)), 121, 456, 789, 101112)
	assert.False(t, ok)
	assert.Equal(t, node128{data1: 123, data2: 456}, *addr)

	ok = compareAndSwapUint128((*uint128)(unsafe.Pointer(addr)), 123, 400, 789, 101112)
	assert.False(t, ok)
	assert.Equal(t, node128{data1: 123, data2: 456}, *addr)

	ok = compareAndSwapUint128((*uint128)(unsafe.Pointer(addr)), 123, 456, 789, 101112)
	assert.True(t, ok)
	assert.Equal(t, node128{data1: 789, data2: 101112}, *addr)
}

func TestCASP(t *testing.T) {
	addr := &node128{data1: 123, data2: 456}
	ok := CASPUint128((*uint128)(unsafe.Pointer(addr)), 121, 456, 789, 101112)
	assert.False(t, ok)
	assert.Equal(t, node128{data1: 123, data2: 456}, *addr)

	ok = CASPUint128((*uint128)(unsafe.Pointer(addr)), 123, 400, 789, 101112)
	assert.False(t, ok)
	assert.Equal(t, node128{data1: 123, data2: 456}, *addr)

	ok = CASPUint128((*uint128)(unsafe.Pointer(addr)), 123, 456, 789, 101112)
	assert.True(t, ok)
	assert.Equal(t, node128{data1: 789, data2: 101112}, *addr)
}

func TestDWCAS(t *testing.T) {
	addr := &uint128{123, 456}
	ok := CASUint128(addr, uint128{121, 456}, uint128{789, 101112})
	assert.False(t, ok)
	assert.Equal(t, uint128{123, 456}, *addr)

	ok = CASUint128(addr, uint128{123, 400}, uint128{789, 101112})
	assert.False(t, ok)
	assert.Equal(t, uint128{123, 456}, *addr)

	ok = CASUint128(addr, uint128{123, 456}, uint128{789, 101112})
	assert.True(t, ok)
	assert.Equal(t, uint128{789, 101112}, *addr)
}

func TestLoadUint128(t *testing.T) {
	addr := &node128{data1: 123, data2: 456}
	data := [8]uint128{}
	var wg sync.WaitGroup
	wg.Add(8)
	for i := 0; i < 8; i++ {
		i := i
		go func() {
			data[i] = loadUint128((*uint128)(unsafe.Pointer(addr)))
			wg.Done()
		}()
	}
	wg.Wait()
	t.Log(data)
}

func TestCasScqNode128(t *testing.T) {
	addr := &scqNode128{flag: 123, data: 456}
	ok := compareAndSwapSCQNodePointer(addr, scqNode128{123, 456}, scqNode128{789, 101112})
	assert.True(t, ok)
	assert.Equal(t, scqNode128{flag: 789, data: 101112}, *addr)
}
