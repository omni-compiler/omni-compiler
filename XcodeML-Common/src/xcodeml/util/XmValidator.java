/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.util;

import java.io.BufferedReader;
import java.io.Reader;
import java.io.StringReader;
import java.net.URL;
import java.util.List;

import javax.xml.XMLConstants;
import javax.xml.transform.stream.StreamSource;
import javax.xml.validation.Schema;
import javax.xml.validation.SchemaFactory;
import javax.xml.validation.Validator;

import org.xml.sax.ErrorHandler;
import org.xml.sax.SAXException;
import org.xml.sax.SAXNotRecognizedException;
import org.xml.sax.SAXNotSupportedException;
import org.xml.sax.SAXParseException;

import xcodeml.XmException;
import xcodeml.binding.XmXcodeProgram;

/**
 * Validator of XcodeML for C.
 */
public class XmValidator
{
    private static final String SCHEMA_LANGUAGE = XMLConstants.W3C_XML_SCHEMA_NS_URI;
    private Schema _schema;
    
    /**
     * Creates validator.
     */
    public XmValidator(URL schemaFile) throws XmException
    {
        SchemaFactory schemaFactory =
            SchemaFactory.newInstance(SCHEMA_LANGUAGE);
        try {
            schemaFactory.setFeature(
                "http://apache.org/xml/features/validation/schema-full-checking",
                false);
        } catch(SAXNotRecognizedException e) {
            throw new XmException(e);
        } catch(SAXNotSupportedException e) {
            throw new XmException(e);
        }

        try {
            _schema =
                schemaFactory.newSchema(schemaFile);
        } catch(SAXException e) {
            throw new XmException(e);
        }
    }

    
    /**
     * Reads XcodeML and creates a binding object and errors
     *
     * @param reader XcodeML input
     * @param xmprog a binding object created
     * @param errorList validate errors stored
     * @return false if validate error has occurred
     */
    public boolean read(Reader reader, XmXcodeProgram xmprog, List<String> errorList)
    {
        String s = null;
        
        try {
            StringBuilder sb = new StringBuilder(1024 * 4);
            BufferedReader br = new BufferedReader(reader);
            String line = null;
            
            while((line = br.readLine()) != null) {
                sb.append(line);
                sb.append("\n");
            }
            
            s = sb.toString();
            sb = null;
            StringReader sreader = new StringReader(s);
            
            StreamSource source = new StreamSource(sreader);
            Validator validator = _schema.newValidator();
            validator.setErrorHandler(new ValidatorErrorHandler(errorList));
            validator.validate(source);
            
            if(errorList.size() == 0) {
                sreader = new StringReader(s);
                xmprog.setup(sreader);
            }
            
        } catch (Exception e) {
            if(e.getMessage() == null) {
                errorList.add(e.getClass().getName());
                e.printStackTrace();
            } else {
                errorList.add(e.getMessage());
            }
        }

        return (errorList.size() == 0);
    }

    static class ValidatorErrorHandler implements ErrorHandler
    {
        private List<String> _errorList;

        public ValidatorErrorHandler(List<String> errorList)
        {
            _errorList = errorList;
        }

        private String _errorMessage(SAXParseException e)
        {
            String m = e.getMessage();
            if(m != null && m.indexOf(":") > 0) {
                m = m.substring(m.indexOf(":") + 1, m.length()).trim();
            }
            return m + " at " + e.getLineNumber() + "," + e.getColumnNumber();
        }

        @Override
        public void error(SAXParseException e) throws SAXException
        {
            _errorList.add(_errorMessage(e));
        }

        @Override
        public void fatalError(SAXParseException e) throws SAXException
        {
            _errorList.add(_errorMessage(e));
        }

        @Override
        public void warning(SAXParseException e) throws SAXException
        {
            _errorList.add(_errorMessage(e));
        }
    }
}
