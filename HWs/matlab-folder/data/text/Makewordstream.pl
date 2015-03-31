#!/usr/bin/perl -w

#----------------------------------------------------
# file:    Makewordstream.pl
# purpose: Produce stream of words from each text file listed in docs.txt
# usage:   Makewordstream.pl
# inputs:  stdin (docs.txt)
# outputs: stdout (wordstream.txt)
# author:  David Newman, newman@uci.edu
#----------------------------------------------------

use strict;

my $word;
my $subword;
my @stopwords = qw(the and was for that you with have are this from can which has were don); # stopwords to always remove
my %isstopword = ();
foreach $word (@stopwords) { $isstopword{$word} = 1 }

#-------------------------------------------
# scan thru all files listed in docs.txt
#-------------------------------------------
while(<>) {
    chomp;
    
    open(F_IN,"< $_") or die "can't open file $_";
    print "### $_\n"; # leave filename in stream for validation
    
    while (<F_IN>) {
	chomp;
	my @allwords = split;         # split on space
	foreach $word (@allwords) {
	    $word = lc($word);        # convert to lowercase
	    $word =~ s/\W/ /g;        # replace all punctuation with space 
	    my @allsubwords = split(/ /, $word);  # split again (if above subs introduced space)
	    foreach $subword (@allsubwords) {
		unless ($isstopword{$subword}) {                      # suppress above stopwords
		    if (length($subword)>2) { print "$subword\n"; }   # only print words longer than 2 chars
		}
	    }
	}
    }
    
    close(F_IN);
    print "###EOF###\n"; # write EOF marker
}
