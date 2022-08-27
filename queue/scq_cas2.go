package queue

import (
	"golang.org/x/sys/cpu"
	"sync/atomic"
	"unsafe"
)

// ScqCas2 is implementation of double-width CAS version of SCQ
// entry: |-|-|------------------|-----------------------|
//        Cycle(62-order bits) + IsSafe(1 bit) + IsUnused(1 bit) + Index(order bits)
type ScqCas2 struct {
	head      uint64
	_         [unsafe.Sizeof(cpu.CacheLinePad{}) - 8]byte
	tail      uint64
	_         [unsafe.Sizeof(cpu.CacheLinePad{}) - 8]byte
	threshold int64
	_         [unsafe.Sizeof(cpu.CacheLinePad{}) - 8]byte
	next      *ScqCas2
	entries   *[scqsize]uint128
}

func NewScqCas2() *ScqCas2 {
	q := &ScqCas2{
		tail:      scqsize,
		head:      scqsize,
		threshold: -1,
		next:      nil,
	}
	q.entries = new([scqsize]uint128)
	for i, _ := range q.entries {
		q.entries[i] = uint128{DWUnused | DWFlagSafe, 0}
	}
	return q
}

const (
	DWUnused         = uint64(1) << 63 // indicate the slot is unused. (\perp)
	DWFlagSafe       = uint64(1) << 62
	DWFlagUnsafe     = uint64(0)
	DWMask           = uint64(0b11) << 62
	DWResetThreshold = 4*int64(qsize) - 1
)

type scqNode128 struct {
	flag uint64
	data uint64
}

func (q *ScqCas2) Enqueue(val uint64) bool {
	if (atomic.LoadUint64(&q.tail) & DWUnused) == DWUnused {
		return false
	}

	if atomic.LoadUint64(&q.tail) >= atomic.LoadUint64(&q.head)+scqsize {
		return false
	}

	var tail, j, tcycle, ecycle, isSafe, isUnused uint64
	var ent uint128
	newEnt := uint128{DWFlagSafe, val}

	for {
		tail = atomic.AddUint64(&q.tail, 1) - 1
		tcycle = (tail & ^(scqsize - 1)) >> (order + 1) // Cycle(T) actually equals (tcycle >> (order+2))
		j = cacheRemap8BSCQRaw(tail)
	EnqueueRELOAD:
		ent = loadUint128(&q.entries[j])
		ecycle = ent[0] & ^DWMask
		isUnused = ent[0] & DWUnused
		isSafe = ent[0] & DWFlagSafe
		if ecycle < tcycle && isUnused == DWUnused && (isSafe == DWFlagSafe || atomic.LoadUint64(&q.head) <= tail) {
			newEnt[0] = tcycle | DWFlagSafe
			if !CASUint128(&q.entries[j], ent, newEnt) {
				goto EnqueueRELOAD
			}
			if atomic.LoadInt64(&q.threshold) != DWResetThreshold {
				atomic.StoreInt64(&q.threshold, DWResetThreshold)
			}
			return true
		}
		if tail+1 >= atomic.LoadUint64(&q.head)+scqsize {
			return false
		}
	}
}

func (q *ScqCas2) Dequeue() (val uint64, ok bool) {
	if atomic.LoadInt64(&q.threshold) < 0 {
		return empty, false
	}
	var head, j, hcycle, ecycle, isSafe, isUnused, tail uint64
	var ent, newEnt uint128
	for {
		head = atomic.AddUint64(&q.head, 1) - 1
		hcycle = (head & ^(scqsize - 1)) >> (order + 1)
		j = cacheRemap8BSCQRaw(head)
	DequeueRELOAD:
		ent = loadUint128(&q.entries[j])
		ecycle = ent[0] & ^DWMask
		isUnused = ent[0] & DWUnused
		isSafe = ent[0] & DWFlagSafe
		//fmt.Printf("head, hcycle, ecycle, eindex, isSafe: (%v, %v, %v, %v, %v)\n", head, hcycle, ecycle, eindex, isSafe)
		if ecycle == hcycle {
			atomicOrUint64(&(q.entries[j][0]), DWUnused)
			return ent[1], true
		}
		if ecycle < hcycle {
			newEnt[0] = ecycle
			newEnt[1] = ent[1]
			if isUnused == DWUnused {
				newEnt[0] = hcycle | isSafe | DWUnused
			}
			if !CASUint128(&q.entries[j], ent, newEnt) {
				goto DequeueRELOAD
			}
		}
		tail = atomic.LoadUint64(&q.tail)
		if tail <= head+1 {
			q.catchup(tail, head+1)
			atomic.AddInt64(&q.threshold, -1)
			return empty, false
		}
		if atomic.AddInt64(&q.threshold, -1)+1 <= 0 { // atomic.Add-delta = FAA
			return empty, false
		}
	}
}

func (q *ScqCas2) catchup(tail, head uint64) {
	for !atomic.CompareAndSwapUint64(&q.tail, tail, head) {
		head = atomic.LoadUint64(&q.head)
		tail = atomic.LoadUint64(&q.tail)
		if tail >= head {
			break
		}
	}
}
