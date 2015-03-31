#!/usr/bin/perl -w

#----------------------------------------------------
# file:    Makedocword.pl
# purpose: determine doc-word-count matrix for document collection in docs.txt
# usage:   Makedocword.pl
# inputs:  stdin (wordstream.txt), vocab.txt
# outputs: stdout (docword.txt)
# author:  David Newman, newman@uci.edu
#----------------------------------------------------

use strict;

my $word;
my %count = ();
my $docID = 1;
my $wordID;
my $cnt;

#------------------------------------
# read in the vocab
#------------------------------------
open(F_IN,"< vocab.txt") or die "can't open vocab.txt";
my %vocab = ();
my $i = 1;
while (<F_IN>) {
    chomp;
    $vocab{$_} = $i++;
}
close(F_IN);

#-------------------------------------------
# scan thru wordstream
#-------------------------------------------
open(F_OUT,"> tmpdocword.txt") or die "can't open tmpdocword.txt";
my $nrow = 0;
my $ncol = 0;
my $nnz = 0;
while (<>) {
    chomp;
    $word = $_;
    if ($word =~ /^\#\#\#EOF\#\#\#$/) { 
	foreach $word (sort keys %count) {
	    if ($vocab{$word}) {
		$wordID = $vocab{$word};
		$cnt    = $count{$word};
		print F_OUT "$docID $wordID $cnt\n";
		if ($nrow < $docID)  { $nrow = $docID; }
		if ($ncol < $wordID) { $ncol = $wordID; }
		$nnz++;
	    }
	}
	%count = ();
	$docID++;
    } else {
	$count{$word}++;
    }
}
close(F_OUT);

#-------------------------------------------
# print header
#-------------------------------------------
open(F_IN,  "< tmpdocword.txt") or die "can't open tmpdocword.txt";
print "$nrow\n";
print "$ncol\n";
print "$nnz\n";
while (<F_IN>) { print; }
close(F_IN);
unlink 'tmpdocword.txt';
