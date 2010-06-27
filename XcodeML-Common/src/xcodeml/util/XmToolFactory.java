/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.util;

import java.lang.reflect.Method;

import xcodeml.XmException;
import xcodeml.XmLanguage;
import xcodeml.binding.XmXcodeProgram;

/**
 * XcodeML decompiler facotry.
 */
public class XmToolFactory
{
    interface ToolProperty
    {
        public String getValidatorClassName();
        public String getDecompilerClassName();
        public String getDecompilerContextClassName();
        public String getXcodeMLFactoryClassName();
        public String getXcodeMLFactoryMethodName();
        public String getXmObjToXobjectTranslatorClassName();
        public String getXobjectToXmObjTranslatorClassName();
    }
    
    class CToolProperty implements ToolProperty
    {
        public String getValidatorClassName()
        {
            return "xcodeml.c.util.XmcValidator";
        }
        
        public String getDecompilerClassName()
        {
            return "xcodeml.c.util.XmcDecompiler";
        }
        
        public String getDecompilerContextClassName()
        {
            return "xcodeml.c.util.XmcDecompilerContext";
        }
        
        public String getXcodeMLFactoryClassName()
        {
            return "xcodeml.c.binding.gen.XcodeML_CFactory";
        }
        
        public String getXcodeMLFactoryMethodName()
        {
            return "createXbcXcodeProgram";
        }
        
        public String getXmObjToXobjectTranslatorClassName()
        {
            return "exc.xcodeml.XmcXmObjToXobjectTranslator";
        }
        
        public String getXobjectToXmObjTranslatorClassName()
        {
            return "exc.xcodeml.XmcXobjectToXmObjTranslator";
        }
    }
    
    class FToolProperty implements ToolProperty
    {
        public String getValidatorClassName()
        {
            return "xcodeml.f.util.XmfValidator";
        }
        
        public String getDecompilerClassName()
        {
            return "xcodeml.f.util.XmfDecompiler";
        }
        
        public String getDecompilerContextClassName()
        {
            return "xcodeml.f.decompile.XmfDecompilerContext";
        }
        
        public String getXcodeMLFactoryClassName()
        {
            return "xcodeml.f.binding.gen.XcodeML_FFactory";
        }
        
        public String getXcodeMLFactoryMethodName()
        {
            return "createXbfXcodeProgram";
        }

        public String getXmObjToXobjectTranslatorClassName()
        {
            return "exc.xcodeml.XmfXmObjToXobjectTranslator";
        }
        
        public String getXobjectToXmObjTranslatorClassName()
        {
            return "exc.xcodeml.XmfXobjectToXmObjTranslator";
        }
    }
    
    private ToolProperty _property;

    /**
     * Construct factory for 'language'
     * 
     * @param language 'C' or 'F'
     */
    public XmToolFactory(String language) throws XmException
    {
        if(language.equals(XmLanguage.C.toString())) {
            _property = new CToolProperty();
        } else if(language.equals(XmLanguage.F.toString())) {
            _property = new FToolProperty();
        } else {
            throw new XmException("unsupported language '" + language + "'.");
        }
    }
    
    /**
     * create XcodeML validator.
     */
    public XmValidator createValidator() throws XmException
    {
        try {
            return (XmValidator)Class.forName(
                _property.getValidatorClassName()).newInstance();
        } catch(Exception e) {
            throw new XmException(e);
        }
    }
    
    /**
     * create XcodeML decompiler.
     */
    public XmDecompiler createDecompiler() throws XmException
    {
        try {
            return (XmDecompiler)Class.forName(
                _property.getDecompilerClassName()).newInstance();
        } catch(Exception e) {
            throw new XmException(e);
        }
    }
    
    /**
     * create XcodeML decompiler context.
     */
    public XmDecompilerContext createDecompilerContext() throws XmException
    {
        try {
            return (XmDecompilerContext)Class.forName(
                _property.getDecompilerContextClassName()).newInstance();
        } catch(Exception e) {
            throw new XmException(e);
        }
    }
    
    /**
     * create XcodeML Program object.
     */
    public XmXcodeProgram createXcodeProgram() throws XmException
    {
        try {
            Class<?> factoryStaticClass = Class.forName(_property.getXcodeMLFactoryClassName());
            Method methodGetFactory = factoryStaticClass.getMethod("getFactory");
            Object factory = methodGetFactory.invoke(null);
            Method methodCreate = factory.getClass().getMethod(_property.getXcodeMLFactoryMethodName());
            return (XmXcodeProgram)methodCreate.invoke(factory);
        } catch(Exception e) {
            throw new XmException(e);
        }
    }

    /**
     * create XmObj to Xobject translator.
     */
    public XmXmObjToXobjectTranslator createXmObjToXobjectTranslator() throws XmException
    {
        try {
            return (XmXmObjToXobjectTranslator)Class.forName(
                _property.getXmObjToXobjectTranslatorClassName()).newInstance();
        } catch(Exception e) {
            throw new XmException(e);
        }
    }

    /**
     * create Xobject to XmObj translator.
     */
    public XmXobjectToXmObjTranslator createXobjectToXmObjTranslator() throws XmException
    {
        try {
            return (XmXobjectToXmObjTranslator)Class.forName(
                _property.getXobjectToXmObjTranslatorClassName()).newInstance();
        } catch(Exception e) {
            throw new XmException(e);
        }
    }
}
