// +build arm64

#include "textflag.h"
#include "funcdata.h"

TEXT ·atomicAnd(SB), NOSPLIT, $0-12
	MOVD	ptr+0(FP), R0
	MOVW	val+8(FP), R1
load_store_loop:
	LDAXRW	(R0), R2
	AND	R1, R2
	STLXRW	R2, (R0), R3
	CBNZ	R3, load_store_loop
	RET

TEXT ·atomicOr(SB), NOSPLIT, $0-12
	MOVD	ptr+0(FP), R0
	MOVW	val+8(FP), R1
load_store_loop:
	LDAXRW	(R0), R2
	ORR	R1, R2
	STLXRW	R2, (R0), R3
	CBNZ	R3, load_store_loop
	RET

TEXT ·atomicAndUint64(SB), NOSPLIT, $0-16
	MOVD	ptr+0(FP), R0
	MOVD	val+8(FP), R1
load_store_loop:
	LDAXR	(R0), R2
	AND	R1, R2
	STLXR	R2, (R0), R3
	CBNZ	R3, load_store_loop
	RET

TEXT ·atomicOrUint64(SB), NOSPLIT, $0-16
	MOVD	ptr+0(FP), R0
	MOVD	val+8(FP), R1
load_store_loop:
	LDAXR	(R0), R2
	ORR	R1, R2
	STLXR	R2, (R0), R3
	CBNZ	R3, load_store_loop
	RET

TEXT ·compareAndSwapUint128(SB), NOSPLIT, $0-41
	MOVD	addr+0(FP), R0
	MOVD	old1+8(FP), R1
	MOVD	old2+16(FP), R2
	MOVD	new1+24(FP), R3
	MOVD	new2+32(FP), R4
load_store_loop:
	LDAXP	(R0), (R5, R6)
	CMP	R1, R5
	BNE ok
	CMP R2, R6
	BNE ok
	STLXP	(R3, R4), (R0), R7
	CBNZ	R7, load_store_loop
	RET
ok:
    CSET	EQ, R0
    MOVB	R0, ret+40(FP)
    RET

TEXT ·CASPUint128(SB), NOSPLIT, $0-41
	MOVD	addr+0(FP), R0
	MOVD	old1+8(FP), R2
	MOVD	old2+16(FP), R3
	MOVD	new1+24(FP), R4
	MOVD	new2+32(FP), R5
	ORR R2, ZR, R6
	ORR R3, ZR, R7
	CASPD (R2, R3), (R0), (R4, R5)
	CMP R2, R6
	BNE ok
	CMP R3, R7
ok:
    CSET	EQ, R0
    MOVB	R0, ret+40(FP)
    RET

TEXT ·loadUint128(SB),NOSPLIT,$0-24
	MOVD	ptr+0(FP), R0
	LDAXP	(R0), (R0, R1)
	MOVD	R0, ret+8(FP)
	MOVD	R1, ret+16(FP)
	RET