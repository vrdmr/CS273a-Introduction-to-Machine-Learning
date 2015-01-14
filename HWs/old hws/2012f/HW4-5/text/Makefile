CC=gcc
CFLAGS=-c -Wall -O3
LDFLAGS=-lm -O3

all: topicmodel printtopics printtopicsindocs

topicmodel: topicmodel.o tlib.o
printtopics: printtopics.o tlib.o
printtopicsindocs: printtopicsindocs.o tlib.o

clean:
	rm -rf *.o topicmodel printtopics printtopicsindocs
