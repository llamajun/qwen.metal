
CFLAGS?=-O2 -g
CC=clang
OBJCFLAGS=-fobjc-arc
LDFLAGS=-framework Foundation -framework Metal -framework MetalKit
EXES=add gemv run

all: $(EXES)

add: add.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

add.o: add.m
	$(CC) $(CFLAGS) $(OBJCFLAGS) -c add.m -o add.o 

gemv: gemv.o llm-metal.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ gemv.o llm-metal.o

gemv.o: gemv.c
	$(CC) $(CFLAGS) -o $@ -c $<

llm-metal.o: llm-metal.m
	$(CC) $(CFLAGS) $(OBJCFLAGS) -o $@ -c $<

run: run.o llm-metal.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ run.o llm-metal.o

run.o: run.c
	$(CC) $(CFLAGS) -o $@ -c $<

chat: run
	./run qwen1.5-0.5b-chat -m chat -i "介绍一下机器学习。"

generate: run
	./run qwen1.5-0.5b-chat -m generate -i "大模型是"

lldb: run
	lldb ./run -- qwen1.5-0.5b-chat -m chat -i "介绍一下机器学习。"

clean:
	rm -f $(EXES) $(EXES:=.o) matmul.o

.PHONY: all clean
