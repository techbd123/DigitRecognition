BINS = testcounting

all: $(BINS)

clean:
	rm -f *.o *.err *.map $(BINS)

testcounting: testcounting.o backprop.o
	$(CC) $(LDFLAGS) -o testcounting testcounting.o backprop.o $(LIBS)
	
testcounting.o: testcounting.c backprop.h
	$(CC) $(CFLAGS) -c testcounting.c

backprop.o: backprop.c backprop.h
	$(CC) $(CFLAGS) -c backprop.c
