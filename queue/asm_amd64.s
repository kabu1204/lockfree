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

TEXT ·compareAndSwapUint128(SB),NOSPLIT,$0
	MOVQ addr+0(FP), R8
	MOVQ old1+8(FP), AX
	MOVQ old2+16(FP), DX
	MOVQ new1+24(FP), BX
	MOVQ new2+32(FP), CX
	LOCK
	CMPXCHG16B (R8)
	SETEQ swapped+40(FP)
	RET

TEXT ·CASPUint128(SB),NOSPLIT,$0
	JMP ·compareAndSwapUint128(SB)

TEXT ·CASUint128(SB),NOSPLIT,$0
	JMP ·compareAndSwapUint128(SB)

TEXT ·loadUint128(SB),NOSPLIT,$0
	MOVQ addr+0(FP), R8
	XORQ AX, AX
	XORQ DX, DX
	XORQ BX, BX
	XORQ CX, CX
	LOCK
	CMPXCHG16B (R8)
	MOVQ AX, val_0+8(FP)
	MOVQ DX, val_1+16(FP)
	RET

TEXT ·loadUint1282(SB),NOSPLIT,$0
    MOVQ addr+0(FP), AX
	VMOVDQU (AX), X0
	VMOVDQU X0, val+8(FP)
	RET

TEXT ·loadSCQNodePointer(SB),NOSPLIT,$0
    JMP ·loadUint128(SB)

TEXT ·runtimeEnableWriteBarrier(SB),NOSPLIT,$0
	MOVL runtime·writeBarrier(SB), AX
	MOVB AX, ret+0(FP)
	RET
