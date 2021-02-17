#
# Copyright (C) 2019 Exein
#
# This is free software, licensed under the GNU General Public License v3.
# See /LICENSE for more information.
#

PROJECT_BINARY_NAME = mle-player
PROJECT_SOURCES = mle-player.cc


all: $(PROJECT_BINARY_NAME)

$(PROJECT_BINARY_NAME):	$(PROJECT_SOURCES)
	$(CXX) -std=c++14 -funwind-tables -I/usr/include/xtensor -I. $(CXXFLAGS) $(PROJECT_SOURCES) -o $(PROJECT_BINARY_NAME) -ltensorflow-lite -lexnl


clean:
	$(RM) -f $(PROJECT_BINARY_NAME)

