XFC     = xmpf90
XRUN    = mpiexec
EXES    = module_test.x module_test0.x arg.x dnodes.x refobj.x 103.x 228.x 310.x 358_result.x 383.x 377.x 392.x 397.x 152.x 018.x add_decl-block.x add_decl-block-mod.x block_shadow_reflect.x block_template_fix.x block_reflect.x block_bcast_async.x block_tasks.x type_bound_procedure_2.x type_bound_procedure_5.x type_bound_procedure_generic_2.x type_bound_procedure_override.x type_bound_procedure_3.x type_bound_procedure_attr.x type_bound_procedure_generic_3.x type_bound_procedure_use_module.x type_bound_procedure_4.x type_bound_procedure.x type_bound_procedure_generic.x submodule_gmove_inout_3.x procedure_declaration_1.x procedure_declaration_2.x procedure_declaration_3.x procedure_declaration_4.x procedure_declaration_5.x procedure_declaration_7.x procedure_declaration_assignment_1.x procedure_declaration_assignment_3.x procedure_declaration_assignment_4.x procedure_declaration_assignment_from_sibling_module_procedure.x procedure_declaration_in_derived_type_2.x procedure_declaration_in_derived_type.x procedure_declaration_initialization.x procedure_declaration_member_initialize.x procedure_declaration_member_initialize_with_contains_function.x procedure_declaration_member_initialize_with_same.x procedure_declaration_separete_type_and_attribute_1.x procedure_declaration_separete_type_and_attribute_2.x use_procedure_decl.x long_array_3.x assign_to_allocatable_integer_array.x assumed_size.x submodule_01.x value_statement.x array_constractor_2.x array_constractor_with_type.x typed_array_constructor.x forall_nest_name.x use_derived_type_and_generics_in_module_2.x specification_expressions.x omp_collapse.x functional.x

all: module_test.x module_test0.x arg.x dnodes.x external.o interface.o xmp_lib.o threadprivate.o refobj.x \
     103.x 223.o 228.x 230.o 321.o 45_save.o 354.o 358_result.x 371.o 375.x 377.x 382.o 261.o 162.o 383.x 384.o 391.o 392.x 393.o 278\ spacefile.x derived.o func.o schedule_default.o task001.o 152.x 397.x double_option.x nargs.o result.o 018.x namelist.o 464.o 467.o 454.o 469.o 310.x 157.o 156.o 154.o add_decl-block.x add_decl-block-mod.x block_shadow_reflect.x block_template_fix.x block_reflect.x threadprivate-block.o block_bcast_async.x block_tasks.x type_bound_procedure_2.x type_bound_procedure_5.x type_bound_procedure_generic_2.x type_bound_procedure_override.x type_bound_procedure_3.x type_bound_procedure_attr.x type_bound_procedure_generic_3.x type_bound_procedure_use_module.x type_bound_procedure_4.x type_bound_procedure.x type_bound_procedure_generic.x submodule_gmove_inout_3.x nest_use_procedure_decl.o procedure_declaration_1.x procedure_declaration_2.x procedure_declaration_3.x procedure_declaration_4.x procedure_declaration_5.x procedure_declaration_7.x procedure_declaration_assignment_1.x procedure_declaration_assignment_3.x procedure_declaration_assignment_4.x procedure_declaration_assignment_from_sibling_module_procedure.x procedure_declaration_in_derived_type_2.x procedure_declaration_in_derived_type.x procedure_declaration_initialization.x procedure_declaration_member_initialize.x procedure_declaration_member_initialize_with_contains_function.x procedure_declaration_member_initialize_with_same.x procedure_declaration_separete_type_and_attribute_1.x procedure_declaration_separete_type_and_attribute_2.x __procedure_decl.o _use_procedure_decl.o use_procedure_decl.x long_array_3.x assign_to_allocatable_integer_array.x assumed_size.x submodule_01.x value_statement.x array_constractor_2.x array_constractor_with_type.x typed_array_constructor.x forall_nest_name.x __derived_type_and_generics_in_module_2.o use_derived_type_and_generics_in_module_2.x specification_expressions.x omp_collapse.x functional.o 381.o 477.o 477_2.o 409.o 355.o 504.o 535.o 536.o #alloc_opt.o

377.x: __omni_tmp__/forall2_F90_in.F90

###     103.x 223.o 228.x 230.o 321.o 45_save.o 358_result.x 373.o 377.x 261.o 162.o 383.x 384.o 391.o 392.x 393.o 278\ spacefile.x derived.o func.o save_desc.x schedule_default.o task001.o 152.x 397.x double_option.x nargs.o
###mpiexec -n 4 ./static_desc.x
###Fatal error in PMPI_Comm_split: Other MPI error, error stack:
###PMPI_Comm_split(495).................: MPI_Comm_split(comm=0xc4000000, color=1, key=1, new_comm=0xa990c0) failed
###PMPI_Comm_split(477).................: 
###MPIR_Comm_split_impl(249)............: 
###MPIR_Get_contextid_sparse_group(1193): Too many communicators (0/2048 free on this process; ignore_id=0)

.PHONY: clean submit showlog cleanlog

.SUFFIXES: .x .o .f90 .F90

.f90.x:
	$(XFC) -o $@ $<

.F90.x:
	$(XFC) -o $@ $<

.f90.o:
	$(XFC) -c $<

.F90.o:
	$(XFC) -c $<

.f.x:
	$(XFC) -o $@ $<

.f.o:
	$(XFC) -c $<

.F.o:
	$(XFC) -c $<

nargs.o: nargs.f90
	$(XFC) -c --Wn"-w" $<

double_option.x: double_option.f90
	$(XFC) -o double_option.x --Wn"-O2 -O2" double_option.f90

task001.o: task001.f90
	$(XFC) --openmp task001.f90 -c

omp_collapse.o: omp_collapse.f90
	$(XFC) -omp omp_collapse.f90 -c

schedule_default.o: schedule_default.F90
	$(XFC) --openmp schedule_default.F90 -c

278\ spacefile.x: 278\ spacefile.f90
	@rm -rf 278\ module
	mkdir 278\ module
	$(XFC) 278\ spacefile.f90 -J278\ module -o 278\ spacefile.x -cpp -DSTRING='"P A S S"'

module_test0.x: module0.o main.o
	$(XFC) -o module_test0.x module0.o main.o

module_test1.o: module_test2.o module_test1.f90
	$(XFC) -c module_test1.f90

module_test.x: module_test2.o module_test1.o
	$(XFC) -o module_test.x module_test2.o module_test1.o

module_test2.o: module_test2.f90
	$(XFC) -c module_test2.f90

module0.o: module0.f90
	rm -rf mod
	mkdir -p mod
	$(XFC) -c module0.f90 -Jmod

external.o: external.f90
	$(XFC) -c external.f90

main.o: main.f90 module0.o
	$(XFC) -c main.f90 -Jmod

arg.x: arg.o
	$(XFC) -o arg.x arg.o

arg.o: arg.f90
	$(XFC) -c arg.f90

dnodes.x: dnodes.o
	$(XFC) -o dnodes.x dnodes.o

dnodes.o: dnodes.f90
	$(XFC) -c dnodes.f90

interface.o: interface.f90
	$(XFC) -c interface.f90

xmp_lib.o: xmp_lib.f90
	$(XFC) -c xmp_lib.f90

threadprivate.o: threadprivate.f90
	$(XFC) -c --openmp threadprivate.f90

threadprivate-block.o: threadprivate-block.F90
	$(XFC) -c --openmp threadprivate-block.F90

refobj.x: refobj.f90
	$(XFC) -o refobj.x refobj.f90

228.x: 228.f
	$(XFC) -o 228.x 228.f

45_save.o: 45_save.f90
	$(XFC) -c 45_save.f90

358_result.x: 358_result.f90
	$(XFC) -o $@ $<

377.x: 377.f90
	$(XFC) -c -o 377.x $<; \
	[ ! -f 377.x ] && exit 1; \
	rm -rf __omni_tmp__; \
	$(XFC) -c $< --debug 1> /dev/null;\
	[ ! -d __omni_tmp__ ] && exit 1; \
        echo "PASS"

261.o: 261.f90
	$(XFC) -c $<

162.o: 162.f90
	$(XFC) -c $<

383.x: 383.f90
	$(XFC) -o $@ $<

392.x: 392.f90
	$(XFC) -I./392-include -o $@ $<

412.x: 412.f90
	$(XFC) -c $< ;\
	if [ ! -f m412.xmod ]; then exit 1; fi

018.x: 018.f90
	$(XFC) -o 018.x $< --tmp ; \
	grep -q "# 9" __omni_tmp__018.f90 || exit 1; \
	echo "PASS"

submodule_02.o: submodule_01.o

submodule_01.x: submodule_01.o submodule_02.o
	$(XFC) -o $@ $^

_use_procedure_decl.o: __procedure_decl.o
nest_use_procedure_decl.o: _use_procedure_decl.o
use_procedure_decl.o: _use_procedure_decl.o

use_procedure_decl.x: __procedure_decl.o _use_procedure_decl.o use_procedure_decl.o
	$(XFC) -o $@ $^

__omni_tmp__/forall2_F90_in.F90: forall2.F90
	$(XFC) --stop-translator --debug $<
	echo "PASS"

use_derived_type_and_generics_in_module_2.o: __derived_type_and_generics_in_module_2.o

use_derived_type_and_generics_in_module_2.x: __derived_type_and_generics_in_module_2.o use_derived_type_and_generics_in_module_2.o
	$(XFC) -o $@ $^

alloc_opt.o: alloc_opt.f90

run:
	$(XRUN) -np 2 ./module_test.x
	$(XRUN) -np 2 ./module_test0.x
	$(XRUN) -np 4 ./arg.x
	$(XRUN) -np 8 ./310.x
	XMP_NODE_SIZE0=2 XMP_NODE_SIZE1=2 $(XRUN) -np 8 ./dnodes.x
	$(XRUN) -np 4 ./refobj.x
	XMP_NODE_SIZE0=2 XMP_NODE_SIZE1=2 $(XRUN) -np 2 ./103.x
	XMP_NODE_SIZE0=2 XMP_NODE_SIZE1=2 $(XRUN) -np 1 ./228.x
	XMP_NODE_SIZE0=2 XMP_NODE_SIZE1=2 $(XRUN) -np 1 ./358_result.x
	XMP_NODE_SIZE0=2 XMP_NODE_SIZE1=2 $(XRUN) -np 2 ./383.x
	XMP_NODE_SIZE0=2 XMP_NODE_SIZE1=2 $(XRUN) -np 1 ./392.x
	XMP_NODE_SIZE0=2 XMP_NODE_SIZE1=2 $(XRUN) -np 1 ./397.x
	XMP_NODE_SIZE0=2 XMP_NODE_SIZE1=2 $(XRUN) -np 1 ./278\ spacefile.x
#	XMP_NODE_SIZE0=2 XMP_NODE_SIZE1=2 $(XRUN) -np 4 ./save_desc.x
	$(XRUN) -np 2 ./add_decl-block.x
	$(XRUN) -np 2 ./add_decl-block-mod.x
	$(XRUN) -np 4 ./block_shadow_reflect.x
	$(XRUN) -np 4 ./block_template_fix.x
	$(XRUN) -np 4 ./block_reflect.x
	$(XRUN) -np 8 ./block_bcast_async.x
	$(XRUN) -np 8 ./block_tasks.x
	$(XRUN) -np 1 ./type_bound_procedure_2.x
	$(XRUN) -np 1 ./type_bound_procedure_5.x
	$(XRUN) -np 1 ./type_bound_procedure_generic_2.x
	$(XRUN) -np 1 ./type_bound_procedure_override.x
	$(XRUN) -np 1 ./type_bound_procedure_3.x
	$(XRUN) -np 1 ./type_bound_procedure_attr.x
	$(XRUN) -np 1 ./type_bound_procedure_generic_3.x
	$(XRUN) -np 1 ./type_bound_procedure_use_module.x
	$(XRUN) -np 1 ./type_bound_procedure_4.x
	$(XRUN) -np 1 ./type_bound_procedure.x
	$(XRUN) -np 1 ./type_bound_procedure_generic.x
	$(XRUN) -np 8 ./submodule_gmove_inout_3.x
	$(XRUN) -np 1 ./submodule_01.x
	$(XRUN) -np 1 ./procedure_declaration_1.x
	$(XRUN) -np 1 ./procedure_declaration_2.x
	$(XRUN) -np 1 ./procedure_declaration_3.x
	$(XRUN) -np 1 ./procedure_declaration_4.x
	$(XRUN) -np 1 ./procedure_declaration_5.x
	$(XRUN) -np 1 ./procedure_declaration_7.x
	$(XRUN) -np 1 ./procedure_declaration_assignment_1.x
	$(XRUN) -np 1 ./procedure_declaration_assignment_3.x
	$(XRUN) -np 1 ./procedure_declaration_assignment_4.x
	$(XRUN) -np 1 ./procedure_declaration_assignment_from_sibling_module_procedure.x
	$(XRUN) -np 1 ./procedure_declaration_in_derived_type_2.x
	$(XRUN) -np 1 ./procedure_declaration_in_derived_type.x
	$(XRUN) -np 1 ./procedure_declaration_initialization.x
	$(XRUN) -np 1 ./procedure_declaration_member_initialize.x
	$(XRUN) -np 1 ./procedure_declaration_member_initialize_with_contains_function.x
	$(XRUN) -np 1 ./procedure_declaration_member_initialize_with_same.x
	$(XRUN) -np 1 ./procedure_declaration_separete_type_and_attribute_1.x
	$(XRUN) -np 1 ./procedure_declaration_separete_type_and_attribute_2.x
	$(XRUN) -np 1 ./use_procedure_decl.x
	$(XRUN) -np 4 ./long_array_3.x
	$(XRUN) -np 1 ./assign_to_allocatable_integer_array.x
	$(XRUN) -np 1 ./assumed_size.x
	$(XRUN) -np 1 ./value_statement.x
	$(XRUN) -np 1 ./array_constractor_2.x
	$(XRUN) -np 1 ./array_constractor_with_type.x
	$(XRUN) -np 1 ./typed_array_constructor.x
	$(XRUN) -np 1 ./forall_nest_name.x
	$(XRUN) -np 1 ./use_derived_type_and_generics_in_module_2.x
	$(XRUN) -np 1 ./specification_expressions.x


RUN:
	mkdir RUN

RUN/%.x:: %.x
	cp $< $@

RUN/go.sh: go.template Makefile
	cp $< $@ && grep XRUN Makefile | sed -e 's/(XRUN)/{XRUN}/' -e 's/ *= */=/' | grep -v Makefile >>$@

submit: all RUN RUN/go.sh $(EXES:%=RUN/%)
	cp 278\ spacefile.x RUN
	cd RUN; pjsub go.sh

showlog:
	cat RUN/go.sh.e* RUN/go.sh.o*

cleanlog:
	rm -rf RUN

clean: cleanlog
	rm -f $(EXES) *.o *.*mod *.s 278\ spacefile.x *.x
	rm -rf mod __omni_tmp__ 278\ module
	rm -f __omni_tmp__*
