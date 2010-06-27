/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;

import xcodeml.XmException;

/**
 * Pragma lexer which processes pragma statement in XcodeML.
 */
public interface ExternalPragmaLexer
{
    public PragmaLexer.Result continueLex() throws XmException;
}
