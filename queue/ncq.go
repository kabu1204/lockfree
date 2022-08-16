package queue

import (
	"golang.org/x/sys/cpu"
	"sync/atomic"
	"unsafe"
)

type entry struct {
	cycle, index uint64
}

type ncq struct {
	head    uint64
	_       [unsafe.Sizeof(cpu.CacheLinePad{})]byte
	tail    uint64
	_       [unsafe.Sizeof(cpu.CacheLinePad{})]byte
	entries [uint64(1) << order]atomic.Value
}

func (q *ncq) InitEmpty() {
	q.head = uint64(1) << order
	q.tail = uint64(1) << order
	for i, _ := range q.entries {
		q.entries[i].Store(entry{cycle: 0, index: 0})
	}
}

func (q *ncq) InitFull() {
	q.head = 0
	q.tail = uint64(1) << order
	for i, _ := range q.entries {
		q.entries[cacheRemap16B(uint64(i))].Store(entry{cycle: 0, index: uint64(i)})
	}
}

func (q *ncq) Enqueue(index uint64) {
	newEnt := entry{index: index}
	var tail, j, tcycle uint64
	var ent entry
	for {
		tail = atomic.LoadUint64(&q.tail)
		j = cacheRemap16B(tail)
		ent = q.entries[j].Load().(entry)
		tcycle = tail & ^(qsize - 1) >> order // tail/n = (tail & ~(n - 1)) >> order
		if ent.cycle == tcycle {
			atomic.CompareAndSwapUint64(&q.tail, tail, tail+1)
			continue
		}
		if ent.cycle+1 != tcycle {
			continue
		}
		newEnt.cycle = tcycle
		if q.entries[j].CompareAndSwap(ent, newEnt) {
			break
		}
	}
	atomic.CompareAndSwapUint64(&q.tail, tail, tail+1)
}

func (q *ncq) Dequeue() (index uint64) {
	var head, j, hcycle uint64
	var ent entry
	for {
		head = atomic.LoadUint64(&q.head)
		j = cacheRemap16B(head)
		ent = q.entries[j].Load().(entry)
		hcycle = head & ^(qsize - 1) >> order // head/n = (head & ~(n - 1)) >> order
		if ent.cycle != hcycle {
			if ent.cycle+1 == hcycle { // wrap around
				return uint64max
			}
			continue
		}
		if atomic.CompareAndSwapUint64(&q.head, head, head+1) {
			break
		}
	}
	return ent.index
}

func NewNCQ() *ncq {
	return &ncq{}
}
