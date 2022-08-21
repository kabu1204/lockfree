// +build amd64

#include "textflag.h"
#include "funcdata.h"

TEXT ·atomicOr(SB), NOSPLIT, $0-12
	MOVQ	ptr+0(FP), AX
	MOVL	val+8(FP), BX
	LOCK
	ORL	BX, (AX)
	RET

TEXT ·atomicAnd(SB), NOSPLIT, $0-12
	MOVQ	ptr+0(FP), AX
	MOVL	val+8(FP), BX
	LOCK
	ANDL	BX, (AX)
	RET

TEXT ·atomicAndUint64(SB), NOSPLIT, $0-16
	MOVQ	ptr+0(FP), AX
	MOVQ	val+8(FP), BX
	LOCK
	ANDQ	BX, (AX)
	RET

TEXT ·atomicOrUint64(SB), NOSPLIT, $0-16
	MOVQ	ptr+0(FP), AX
	MOVQ	val+8(FP), BX
	LOCK
	ORQ	    BX, (AX)
	RET
