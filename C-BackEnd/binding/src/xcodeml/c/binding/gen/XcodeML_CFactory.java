/*
 * The Relaxer artifact
 * Copyright (c) 2000-2003, ASAMI Tomoharu, All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * - Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer. 
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
package xcodeml.c.binding.gen;

import xcodeml.binding.*;

/**
 * XcodeML_CFactory is generated by Relaxer based on XcodeML_C.rng.
 *
 * @version XcodeML_C.rng 1.0 (Thu Sep 24 16:30:21 JST 2009)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XcodeML_CFactory {
    private static IXcodeML_CFactory factory;

    /**
     * Sets a factory.
     *
     * @param newFactory
     */
    public static void setFactory(IXcodeML_CFactory newFactory) {
        factory = newFactory;
    }

    /**
     * Gets the factory.
     *
     * @return IXcodeML_CFactory
     */
    public static IXcodeML_CFactory getFactory() {
        if (factory == null) {
            factory = new DefaultXcodeML_CFactory();
        }
        return (factory);
    }
}
