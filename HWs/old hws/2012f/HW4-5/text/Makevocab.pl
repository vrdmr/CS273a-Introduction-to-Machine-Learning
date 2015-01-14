#!/usr/bin/perl -w

#----------------------------------------------------
# file:    Makevocab.pl
# purpose: determine list of unique words in wordstream.txt
# usage:   Makevocab.pl
# inputs:  stdin (wordstream.txt), and stopwords.txt
# outputs: stdout (vocab.txt) and freqwords.txt
# author:  David Newman, newman@uci.edu
#----------------------------------------------------

use strict;

#------------------------------------
# read in the list of stopwords
#------------------------------------
my %isstopword = ();
open(F_IN, "< stopwords.txt") or die "can't open stopwords.txt";
while (<F_IN>) {
    chomp;
    $isstopword{$_} = 1;
}
close(F_IN);

#-------------------------------------------
# scan thru wordstream.txt
#-------------------------------------------
my $word;
my %seen = ();
while (<>) {
    chomp;
    $word = $_;
    if ($word !~ /^\#\#\#/) { # ignore filename and EOF lines
	unless ($isstopword{$word}) { $seen{$word}++; }
    }
}

#-------------------------------------------
# keep words that occur more than mincount times
#-------------------------------------------
my %seenmany = ();
my $count;
my $mincount = 9;
while ( ($word, $count) = each %seen ) {
    if ($count > $mincount) {
	$seenmany{$word} = $count;
    }
}

#-------------------------------------------
# print sorted vocab to stdout (vocab.txt)
#-------------------------------------------
foreach $word (sort keys(%seenmany)) {
    print "$word\n";
}

#-------------------------------------------
# print frequent words to freqwords.txt
#-------------------------------------------
open(F_OUT,"> freqwords.txt") or die "can't open freqwords.txt";
my %bucket = ();
while ( ($word, $count) = each %seenmany ) {
    push(@{$bucket{$count}},$word);
}

foreach $count (sort { $b <=> $a } keys(%bucket)) {
    foreach (sort { $b cmp $a } @{$bucket{$count}}) {
        printf(F_OUT "%9d %s\n", $count, $_);
    }
}
close(F_OUT);
