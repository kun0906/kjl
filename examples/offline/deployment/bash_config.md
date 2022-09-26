https://www.gnu.org/software/autoconf/manual/autoconf-2.69/html_node/Package-Options.html

15.3 Choosing Package Options
If a software package has optional compile-time features, the user can give configure command line options to specify whether to compile them. The options have one of these forms:
--enable-feature[=arg]
--disable-feature
These options allow users to choose which optional features to build and install. --enable-feature options should never make a feature behave differently or cause one feature to replace another. They should only cause parts of the program to be built rather than left out.

The user can give an argument by following the feature name with ‘=’ and the argument. Giving an argument of ‘no’ requests that the feature not be made available. A feature with an argument looks like --enable-debug=stabs.

**If no argument is given, it defaults to ‘yes’. --disable-feature is equivalent to --enable-feature=no.**

Normally configure scripts complain about --enable-package options that they do not support. See Option Checking, for details, and for how to override the defaults.

For each optional feature, configure.ac should call AC_ARG_ENABLE to detect whether the configure user asked to include it. Whether each feature is included or not by default, and which arguments are valid, is up to you.

— Macro: AC_ARG_ENABLE (feature, help-string, [action-if-given], [action-if-not-given])
If the user gave configure the option --enable-feature or --disable-feature, run shell commands action-if-given. If neither option was given, run shell commands action-if-not-given. The name feature indicates an optional user-level facility. It should consist only of alphanumeric characters, dashes, plus signs, and dots.

The option's argument is available to the shell commands action-if-given in the shell variable enableval, which is actually just the value of the shell variable named enable_feature, with any non-alphanumeric characters in feature changed into ‘_’. You may use that variable instead, if you wish. The help-string argument is like that of AC_ARG_WITH (see External Software).

You should format your help-string with the macro AS_HELP_STRING (see Pretty Help Strings).

See the examples suggested with the definition of AC_ARG_WITH (see External Software) to get an idea of possible applications of AC_ARG_ENABLE.

https://stackoverflow.com/questions/8059196/what-does-variableset-mean

t took be some time, but I found a link explaining what this does. It is a form of bash parameter-substitution that will evaluate to "set" if $VARIABLE has been set and null otherwise. This allows you to check if a variable is set by doing the following:

if [ -z "${VARIABLE+set}" ] ; then
echo "VARIABLE is not set"
fi
It is also interesting to note that ${VARIABLE+set} can just as easily be ${VARIABLE+anything}. The only reason for using +set is because it is slightly more self-documenting (although not enough to keep me from asking this question).

https://devmanual.gentoo.org/general-concepts/autotools/index.html
What is configure AC?
configure.ac (sometimes also named: configure.in) is an input file for autoconf. It contains tests that check for conditions that are likely to differ on different platforms. The tests are made by actually invoke autoconf macros.
http://www.adp-gmbh.ch/misc/tools/configure/configure_in.html#:~:text=configure.ac%20(sometimes%20also%20named,by%20actually%20invoke%20autoconf%20macros.

So far we've only seen 'hard' dependencies. Many packages have optional support for various extras (graphics toolkits, libraries which add functionality, interpreters, features, ...). This is (if we're lucky) handled via --enable-foo and --disable-foo switches to ./configure. which are generated from autoconf rules.

A simple --enable / --disable function might look something like the following:

AC_MSG_CHECKING(--enable-cscope argument)
AC_ARG_ENABLE(cscope,
[  --enable-cscope         Include cscope interface.],
[enable_cscope=$enableval],
[enable_cscope="no"])
AC_MSG_RESULT($enable_cscope)
if test "$enable_cscope" = "yes"; then
AC_DEFINE(FEAT_CSCOPE)
fi

https://pubs.opengroup.org/onlinepubs/9699919799/utilities/V3_chap02.html#tag_18_06_02

${parameter+word}

substitute word

substitute word

substitute null

https://stackoverflow.com/questions/8411616/how-to-list-features-that-can-be-enabled-and-disabled-in-configure-script

./configure --help will do the trick.

This shows any --enable-X or --with-X arguments that are defined using the macros AC_ARG_ENABLE or AC_ARG_WITH, as well as a list of environment variables that the configure script will pay attention to, such as CC.

In some large projects that are organized as a series of sub-projects each with their own configure script, you may need to do ./configure --help=recursive to see all the features/packages for all the sub-projects.

AFAIK, if configure.ac uses the AC_ARG_ENABLE and AC_ARG_WITH macros, the options should show up in the help output. I don't know why a package would try to circumvent this, unless it's an old script. A search of the configure.ac or configure.in script for these macros might help.
