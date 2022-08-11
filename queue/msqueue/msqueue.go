// Package msqueue implements Michael & Scott lock-free queue algorithm
// Find the paper at: https://www.cs.rochester.edu/~scott/papers/1996_PODC_queues.pdf
package msqueue

import (
	"sync"
	"sync/atomic"
)

type pointer struct {
	ptr   *node
	count uint64
}

type node struct {
	value                      interface{}
	next                       atomic.Value
}

type queue struct {
	head                       atomic.Value
	p1, p2, p3, p4, p5, p6, p7 int64
	tail                       atomic.Value
	pool                       sync.Pool
}

//func newAtomicPointer(ptr *node) atomic.Value {
//
//}

func NewMSQueue() *queue {
	var p atomic.Value
	p.Store(pointer{ptr: nil})
	nd := node{next: p, value: nil}
	var ap atomic.Value
	ap.Store(pointer{ptr: &nd})
	return &queue{
		head: ap,
		tail: ap,
	}
}

func (q *queue) enque(value interface{}) {
	var ap atomic.Value // use Pointer?
	var tail, next pointer
	ap.Store(pointer{ptr: nil})
	nd := node{value: value, next: ap}
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

func (q *queue) deque() (ret interface{}, ok bool) {
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
