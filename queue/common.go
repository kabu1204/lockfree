package queue

import (
	"golang.org/x/sys/cpu"
	"unsafe"
)

const (
	cacheLineSize = uint64(unsafe.Sizeof(cpu.CacheLinePad{}))
)

const (
	order             = 16
	uint64max         = ^(uint64(0))
	qsize             = uint64(1) << order // namely n in paper
	scqsize           = qsize << 1         // 2n
	cacheBlockSize8B  = cacheLineSize / 8
	cacheBlockSize16B = cacheLineSize / 16
)

// CacheRemap16B map adjacent indices to different cache lines
// to avoid false-sharing problem.
// example: RawIndices of a and b are 2(line 0 col 2) and 3(line 0 col 3), remapped indices
// are 16(line 2 col 0) and 24(line 3 col 0)
//
//       0     1     2     3     4     5     6     7
// 0 |      |     |  a  |  b  |     |     |     |     |
// 1 |      |     |     |     |     |     |     |     |
// 2 |  a_  |     |     |     |     |     |     |     |
// 3 |  b_  |     |     |     |     |     |     |     |
// 4 |      |     |     |     |     |     |     |     |
// 5 |      |     |     |     |     |     |     |     |
// 6 |      |     |     |     |     |     |     |     |
func cacheRemap16B(index uint64) uint64 {
	rawIndex := index & (qsize - 1)
	cacheLineNum := (rawIndex) % (qsize / cacheBlockSize16B)
	cacheLineIdx := rawIndex / (qsize / cacheBlockSize16B)
	return cacheLineNum*cacheBlockSize16B + cacheLineIdx
}

func cacheRemap8B(index uint64) uint64 {
	rawIndex := index & (qsize - 1)
	cacheLineNum := (rawIndex) % (qsize / cacheBlockSize8B)
	cacheLineIdx := rawIndex / (qsize / cacheBlockSize8B)
	return cacheLineNum*cacheBlockSize8B + cacheLineIdx
}

func cacheRemap8BSCQRaw(index uint64) uint64 {
	rawIndex := index & (scqsize - 1)
	return (rawIndex >> (order - 2)) | ((index << 3) & (scqsize - 1))
}

func cacheRemap8BSCQ(index uint64) uint64 {
	rawIndex := index & (qsize - 1)
	return (rawIndex >> (order - 3)) | ((index << 3) & (qsize - 1))
}
