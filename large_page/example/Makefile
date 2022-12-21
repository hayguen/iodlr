CPPFLAGS=-O3 -std=c++11 -z noexecstack -z relro -z now -Wformat -Wformat-security -Wall
OBJDIR=$(shell realpath obj)
OBJS = $(addprefix $(OBJDIR)/,large_page_example.o)
LDFLAGS = -Wl,-T ../ld.implicit.script
# -lstdc++

.PHONY: all
all: large_page_example

LARGE_PAGE_EXAMPLE_DEPS=    \
	$(OBJS)                   \
	$(OBJDIR)/liblarge_page.a \

large_page_example: $(LARGE_PAGE_EXAMPLE_DEPS)
	@g++ $(LARGE_PAGE_EXAMPLE_DEPS) $(LDFLAGS) -o $@

$(OBJDIR)/liblarge_page.a:
	$(MAKE) -C .. OUTDIR=$(OBJDIR)

$(OBJDIR)/%.o : %.cc
	@g++ $(CPPFLAGS) -o $@ -c -I.. $<

$(OBJS): | $(OBJDIR)

$(OBJDIR):
	@mkdir -p $(OBJDIR)

clean:
	$(MAKE) -C .. OUTDIR=$(OBJDIR) clean
	@rm -rf $(OBJDIR) large_page_example
