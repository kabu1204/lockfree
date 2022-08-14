package lockfree

import (
	"golang.org/x/sys/cpu"
	"unsafe"
)

const (
	defaultOrder  = 4
	defaultSize   = 1 << defaultOrder
	CacheLineSize = unsafe.Sizeof(cpu.CacheLinePad{})
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
func CacheRemap16B(index uint64) uint64 {
	const cacheLineSize = CacheLineSize / 16
	rawIndex := index & uint64(defaultSize-1)
	cacheLineNum := (rawIndex) % (defaultSize / uint64(cacheLineSize))
	cacheLineIdx := rawIndex / (defaultSize / uint64(cacheLineSize))
	return cacheLineNum*uint64(cacheLineSize) + cacheLineIdx
}
