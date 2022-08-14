// Package msqueue implements Michael & Scott lock-free msqueue algorithm
// Find the paper at: https://www.cs.rochester.edu/~scott/papers/1996_PODC_queues.pdf
package queue

import (
	"sync"
	"sync/atomic"
)

type pointer struct {
	ptr   *node
	count uint64
}

type node struct {
	value interface{}
	next  atomic.Value
}

type msqueue struct {
	head                       atomic.Value
	p1, p2, p3, p4, p5, p6, p7 int64
	tail                       atomic.Value
	pool                       *sync.Pool
}

//func newAtomicPointer(ptr *node) atomic.Value {
//
//}

var atomicNilPointer atomic.Value

func init() {
	atomicNilPointer.Store(pointer{ptr: nil})
}

func NewMSQueue() *msqueue {
	pool := &sync.Pool{New: func() interface{} {
		nd := node{next: atomicNilPointer, value: nil}
		return nd
	}}
	var ap atomic.Value
	nd := pool.Get().(node)
	ap.Store(pointer{ptr: &nd})
	return &msqueue{
		head: ap,
		tail: ap,
		pool: pool,
	}
}

func (q *msqueue) EnquePool(value interface{}) {
	var tail, next pointer
	nd := q.pool.Get().(node)
	nd.value = value
	nd.next = atomicNilPointer
	p := pointer{ptr: &nd}
	for {
		tail = q.tail.Load().(pointer)
		next = tail.ptr.next.Load().(pointer)
		if tail == q.tail.Load().(pointer) {
			if next.ptr == nil {
				p.count = next.count + 1
				if tail.ptr.next.CompareAndSwap(next, p) {
					break
				}
			} else {
				q.tail.CompareAndSwap(tail, pointer{ptr: next.ptr, count: tail.count + 1})
			}
		}
	}
	q.tail.CompareAndSwap(tail, p)
}

func (q *msqueue) DequePool() (ret interface{}, ok bool) {
	var head, tail, next pointer
	ok = false
	for {
		head = q.head.Load().(pointer)
		tail = q.tail.Load().(pointer)
		next = head.ptr.next.Load().(pointer)
		if head == q.head.Load().(pointer) {
			if head.ptr == tail.ptr {
				if next.ptr == nil {
					return nil, false
				}
				q.tail.CompareAndSwap(tail, pointer{ptr: next.ptr, count: tail.count + 1})
			} else {
				ret = next.ptr.value
				if q.head.CompareAndSwap(head, pointer{ptr: next.ptr, count: head.count + 1}) {
					break
				}
			}
		}
	}
	q.pool.Put(*head.ptr)
	ok = true
	return
}

func (q *msqueue) Enque(value interface{}) {
	var tail, next pointer
	nd := node{value: value, next: atomicNilPointer}
	p := pointer{ptr: &nd}
	for {
		tail = q.tail.Load().(pointer)
		next = tail.ptr.next.Load().(pointer)
		if tail == q.tail.Load().(pointer) {
			if next.ptr == nil {
				p.count = next.count + 1
				if tail.ptr.next.CompareAndSwap(next, p) {
					break
				}
			} else {
				q.tail.CompareAndSwap(tail, pointer{ptr: next.ptr, count: tail.count + 1})
			}
		}
	}
	q.tail.CompareAndSwap(tail, p)
}

func (q *msqueue) Deque() (ret interface{}, ok bool) {
	var head, tail, next pointer
	ok = false
	for {
		head = q.head.Load().(pointer)
		tail = q.tail.Load().(pointer)
		next = head.ptr.next.Load().(pointer)
		if head == q.head.Load().(pointer) {
			if head.ptr == tail.ptr {
				if next.ptr == nil {
					return nil, false
				}
				q.tail.CompareAndSwap(tail, pointer{ptr: next.ptr, count: tail.count + 1})
			} else {
				ret = next.ptr.value
				if q.head.CompareAndSwap(head, pointer{ptr: next.ptr, count: head.count + 1}) {
					break
				}
			}
		}
	}
	ok = true
	return
}
