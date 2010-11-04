use inc::Module::Install;

all_from 'lib/Algorithm/KernelKMeans/XS.pm';

can_cc or die "This module requires a C compiler";

ppport;
makemaker_args(
    CC => '$(CXX)',
    OBJECT => '$(O_FILES)',
    XSOPT => '-C++ -hiertype',
    clean => {
        FILES => q{
            *.stackdump
            *.gcov *.gcda *.gcno
            *.out
            nytprof
            cover_db
        },
    },
);

tests 't/*.t';
author_tests 'xt';

requires 'Algorithm::KernelKMeans' => '0.02';

test_requires 'ExtUtils::testlib' => '6.56';
test_requires 'Test::Exception'   => '0.29';
test_requires 'Test::More'        => '0.88';

use_test_base;
auto_set_repository;
auto_include;
WriteAll;