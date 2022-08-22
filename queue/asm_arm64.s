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
	LDAXRD	(R0), R2
	AND	R1, R2
	STLXRD	R2, (R0), R3
	CBNZ	R3, load_store_loop
	RET

TEXT ·atomicOrUint64(SB), NOSPLIT, $0-16
	MOVD	ptr+0(FP), R0
	MOVD	val+8(FP), R1
load_store_loop:
	LDAXRD	(R0), R2
	ORR	R1, R2
	STLXRD	R2, (R0), R3
	CBNZ	R3, load_store_loop
	RET
