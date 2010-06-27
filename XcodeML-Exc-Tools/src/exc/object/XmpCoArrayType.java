/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;

public class XmpCoArrayType extends ArrayType
{
    public XmpCoArrayType(String id, Xtype ref,
        int typeQualFlags, int dim, Xobject asize)
    {
        super(Xtype.XMP_CO_ARRAY, id, ref, typeQualFlags, dim, asize, null);
    }
}
