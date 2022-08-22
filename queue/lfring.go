package queue

type lfring struct {
	data [qsize]uint64
	q    *scq
}

func (ring *lfring) Enqueue(val uint64) {

}

func (ring *lfring) Dequeue() {

}
