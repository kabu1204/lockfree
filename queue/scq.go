package queue

import (
	"fmt"
	"golang.org/x/sys/cpu"
	"sync/atomic"
	"unsafe"
)

// scq is implementation of SCQ
// entry: |---------------|-|-----|
//        Cycle(62-order bits) + IsSafe(1 bit) + IsUnused(1 bit) + Index(order bits)
type scq struct {
	head      uint64
	_         [unsafe.Sizeof(cpu.CacheLinePad{})]byte
	tail      uint64
	_         [unsafe.Sizeof(cpu.CacheLinePad{})]byte
	threshold int64
	entries   [qsize << 1]uint64
}

// unused : 000000011111

const (
	unused         = (qsize << 1) - 1 // indicate the slot is unused. (\perp)
	flagSafe       = uint64(1) << (order + 1)
	flagUnsafe     = uint64(0)
	mask           = flagSafe | unused
	resetThreshold = 3*int64(qsize) - 1
	empty          = uint64max
)

func NewSCQ() *scq {
	return &scq{}
}

func (q *scq) InitEmpty() {
	q.head = scqsize // 2n
	q.tail = scqsize // 2n
	q.threshold = -1
	for i, _ := range q.entries {
		q.entries[i] = mask
	}
}

func (q *scq) InitFull() {
	q.head = scqsize
	q.tail = scqsize + qsize // n
	q.threshold = 3*int64(qsize) - 1
	var i uint64
	for i = 0; i < qsize; i++ {
		q.entries[cacheRemap8BSCQRaw(i)] = (1 << (order + 2)) | flagSafe | i
	}
	for ; i < scqsize; i++ {
		q.entries[cacheRemap8BSCQRaw(i)] = mask
	}
}

func PrintEntry(ent uint64) {
	ecycle := (ent & ^mask) >> 1
	eindex := ent & unused
	isSafe := ent & flagSafe
	fmt.Printf("ecycle, eindex, isSafe = (%v,\t%v,\t%v)\n", ecycle, eindex, isSafe == flagSafe)
}

func (q *scq) Enqueue(index uint64) {
	var tail, j, tcycle, ent, ecycle, eindex, isSafe, newEnt uint64
	for {
		tail = atomic.AddUint64(&q.tail, 1) - 1
		tcycle = tail & ^(scqsize - 1) // Cycle(T) actually equals (tcycle >> (order+2))
		j = cacheRemap8BSCQRaw(tail)
	EnqueueRELOAD:
		ent = atomic.LoadUint64(&q.entries[j])
		ecycle = (ent & ^mask) >> 1
		eindex = ent & unused
		isSafe = ent & flagSafe
		if ecycle < tcycle && eindex == unused && (isSafe == flagSafe || atomic.LoadUint64(&q.head) <= tail) {
			newEnt = (tcycle << 1) | flagSafe | index
			if !atomic.CompareAndSwapUint64(&q.entries[j], ent, newEnt) {
				goto EnqueueRELOAD
			}
			if atomic.LoadInt64(&q.threshold) != resetThreshold {
				atomic.StoreInt64(&q.threshold, resetThreshold)
			}
			return
		}
	}
}

func (q *scq) Dequeue() (index uint64) {
	if atomic.LoadInt64(&q.threshold) < 0 {
		return empty
	}
	var head, j, hcycle, ent, ecycle, eindex, isSafe, newEnt, tail uint64
	for {
		head = atomic.AddUint64(&q.head, 1) - 1
		hcycle = head & ^(scqsize - 1)
		j = cacheRemap8BSCQRaw(head)
	DequeueRELOAD:
		ent = atomic.LoadUint64(&q.entries[j])
		ecycle = (ent & ^mask) >> 1
		eindex = ent & unused
		isSafe = ent & flagSafe
		//fmt.Printf("head, hcycle, ecycle, eindex, isSafe: (%v, %v, %v, %v, %v)\n", head, hcycle, ecycle, eindex, isSafe)
		if ecycle == hcycle {
			atomicOrUint64(&(q.entries[j]), unused)
			return eindex
		}
		if ecycle < hcycle {
			newEnt = (ecycle << 1) | eindex
			if eindex == unused {
				newEnt = (hcycle << 1) | isSafe | unused
			}
			if !atomic.CompareAndSwapUint64(&q.entries[j], ent, newEnt) {
				goto DequeueRELOAD
			}
		}
		tail = atomic.LoadUint64(&q.tail)
		if tail <= head+1 {
			q.catchup(tail, head+1)
			atomic.AddInt64(&q.threshold, -1)
			return empty
		}
		if atomic.AddInt64(&q.threshold, -1) < 0 { // atomic.Add-delta = FAA
			return empty
		}
	}
}

func (q *scq) catchup(tail, head uint64) {
	for !atomic.CompareAndSwapUint64(&q.tail, tail, head) {
		head = atomic.LoadUint64(&q.head)
		tail = atomic.LoadUint64(&q.tail)
		if tail >= head {
			break
		}
	}
}
