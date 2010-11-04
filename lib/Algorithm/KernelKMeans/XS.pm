package Algorithm::KernelKMeans::XS;

use 5.010;
use strict;
use warnings;
our $VERSION = '0.01';

require XSLoader;
XSLoader::load(__PACKAGE__, $VERSION);

1;
__END__

=head1 NAME

Algorithm::KernelKMeans::XS - Drop-in replacement for Algorithm::KernelKMeans

=head1 SYNOPSIS

  use Algorithm::KernelKMeans::XS;

=head1 DESCRIPTION

This module is XS version of L<Algorithm::KernelKMeans>, a implementation of weigheted kernel k-means vector clusterer.

Note that when this module is installed, C<Algorithm::KernelKMeans> behaves as a child class of this.

=head1 AUTHOR

Koichi SATOH E<lt>sekia@cpan.orgE<gt>

=head1 SEE ALSO

L<Algorithm::KernelKMeans>

=head1 LICENSE

The MIT License

Copyright (C) 2010 by Koichi SATOH

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

=cut
