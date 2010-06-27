/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.f.decompile;

/**
 * Error expression in decompiler.
 */
enum XfError
{
    SUCCESS
    {
        @Override
        public String message()
        {
            return "Success.";
        }

        @Override
        public boolean isError()
        {
            return false;
        }

        @Override
        public String format(Object... args)
        {
            assert (args.length == 0);
            return message();
        }
    },

    XCODEML_TYPE_NOT_FOUND
    {
        @Override
        public String message()
        {
            return "Type definition for '%1$s' is not found in type table.";
        }

        /**
         * @param args args[0]: Type name.
         */
        @Override
        public String format(Object... args)
        {
            assert (args.length == 1);
            assert (args[0] instanceof String);
            return String.format(message(), args);
        }
    },

    XCODEML_NAME_NOT_FOUND
    {
        @Override
        public String message()
        {
            return "name '%1$s' is not found in symbol table.";
        }

        /**
         * @param args args[0]: Symbol name.
         */
        @Override
        public String format(Object... args)
        {
            assert (args.length == 1);
            assert (args[0] instanceof String);
            return String.format(message(), args);
        }
    },

    XCODEML_TYPE_MISMATCH
    {
        @Override
        public String message()
        {
            return "Reference type of '%1$s' is defined as '%2$s', but it must be '%3$s'.";
        }

        /**
         * @param args args[0]: Type name. (basic, function, struct, Fint, etc...)
         * @param args args[1]: Actual type. (basic, function, struct, etc...)
         * @param args args[2]: Expect type. (basic, function, struct, etc...)
         */
        @Override
        public String format(Object... args)
        {
            assert (args.length == 3);
            assert (args[0] instanceof String);
            assert (args[1] instanceof String);
            assert (args[2] instanceof String);
            return String.format(message(), args);
        }
    },

    XCODEML_NEED_ATTR
    {
        @Override
        public String message()
        {
            return "The necessary '%1$s' attribute for '%2$s' " +
                   "element of this context does not exist or is empty.";
        }

        /**
         * @param args args[0]: Attribute name.
         * @param args args[1]: Element name.
         */
        @Override
        public String format(Object... args)
        {
            assert (args.length == 2);
            assert (args[0] instanceof String);
            assert (args[1] instanceof String);
            return String.format(message(), args);
        }
    },

    XCODEML_SEMANTICS
    {
        @Override
        public String message()
        {
            return "Detected a semantic error of XcodeML/F during handling of '%1$s' element.";
        }

        /**
         * @param args args[0]: Element name.
         */
        @Override
        public String format(Object... args)
        {
            assert (args.length == 1);
            assert (args[0] instanceof String);
            return String.format(message(), args);
        }
    },

    XCODEML_CYCLIC_TYPE
    {
        @Override
        public String message()
        {
            return "Type of '%1$s' has a cyclic type definition.";
        }

        /**
         * @param args args[0]: Type name. (basic, function, struct, Fint, etc...)
         */
        @Override
        public String format(Object... args)
        {
            assert (args.length == 1);
            assert (args[0] instanceof String);
            return String.format(message(), args);
        }
    },
    ;

    public abstract String message();

    public abstract String format(Object... args);

    public boolean isError()
    {
        return true;
    }
}
