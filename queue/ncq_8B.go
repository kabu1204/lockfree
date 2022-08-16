package queue

import (
	"github.com/kabu1204/lockfree"
	"sync/atomic"
)

type ncq8b struct {
	head, tail uint64
	n          uint64
	entries    [uint64(1) << order]uint64
}

func (q *ncq8b) InitEmpty() {
	q.n = uint64(1) << order
	q.head = uint64(1) << order
	q.tail = uint64(1) << order
	for i, _ := range q.entries {
		q.entries[i] = 0 // cycle=0 + index=0
	}
}

func (q *ncq8b) InitFull() {
	q.head = 0
	q.tail = uint64(1) << order
	q.n = uint64(1) << order
	for i, _ := range q.entries {
		q.entries[lockfree.CacheRemap16B(uint64(i))] = uint64(i)
	}
}

func (q *ncq8b) Enqueue(index uint64) {
	var tail, j, tcycle, ent, ecycle, newEnt uint64
	for {
		tail = atomic.LoadUint64(&q.tail)
		j = lockfree.CacheRemap16B(tail)
		ent = atomic.LoadUint64(&q.entries[j])
		tcycle = tail & ^(q.n - 1) >> order // tail/n = (tail & ~(n - 1)) >> order
		ecycle = ent & ^(q.n - 1) >> order
		if ecycle == tcycle {
			atomic.CompareAndSwapUint64(&q.tail, tail, tail+1)
			continue
		}
		if ecycle+1 != tcycle {
			continue
		}
		newEnt = (tcycle << order) + index
		if atomic.CompareAndSwapUint64(&q.entries[j], ent, newEnt) {
			break
		}
	}
	atomic.CompareAndSwapUint64(&q.tail, tail, tail+1)
}

func (q *ncq8b) Dequeue() (index uint64) {
	var head, j, hcycle, ent, ecycle uint64
	for {
		head = atomic.LoadUint64(&q.head)
		j = lockfree.CacheRemap16B(head)
		ent = atomic.LoadUint64(&q.entries[j])
		hcycle = head & ^(q.n - 1) >> order // head/n = (head & ~(n - 1)) >> order
		ecycle = ent & ^(q.n - 1) >> order
		if ecycle != hcycle {
			if ecycle+1 == hcycle { // wrap around
				return uint64max
			}
			continue
		}
		if atomic.CompareAndSwapUint64(&q.head, head, head+1) {
			break
		}
	}
	return ent & (q.n - 1)
}

func NewNCQ8b() *ncq8b {
	return &ncq8b{}
}
