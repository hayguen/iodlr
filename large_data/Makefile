OUTDIR?=.
CC=g++
CFLAGS=-O3 -std=c++11 -z noexecstack -z relro -z now -Wformat -Wformat-security -Wall
RM=/bin/rm

TARGET = $(OUTDIR)/data-large-reference

.PHONY: all
all: $(TARGET)

$(TARGET): $(TARGET).cc
	$(CC) $(CFLAGS) -o $(TARGET) $(TARGET).cc

.PHONY: clean
clean:
	$(RM) -f *.o $(TARGET)
