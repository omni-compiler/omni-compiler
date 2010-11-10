/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

public class XMPuniqueName {
  private final static int BUFFER_LEN = 32;
  private static XMPuniqueName _tempInstance = null;

  private char[] _buffer;
  private int _pointer;

  private XMPuniqueName() {
    _buffer = new char[BUFFER_LEN];

    _pointer = 0;
    _buffer[0] = 'a';
  }

  private void changeStatus() throws XMPexception {
    changeStatus(0);
  }

  private void changeStatus(int p) throws XMPexception {
    if (p > _pointer) {
      _pointer++;
      if (_pointer == BUFFER_LEN)
        throw new XMPexception("cannot create a unique name");
      else {
        _buffer[_pointer] = 'a';
        return;
      }
    }

    if ((_buffer[p] >= 'a') && (_buffer[p] <= 'y')) {
      _buffer[p]++;
    }
    else if ((_buffer[p] >= 'A') && (_buffer[p] <= 'Y')) {
      _buffer[p]++;
    }
    else if ((_buffer[p] >= '0') && (_buffer[p] <= '8')) {
      _buffer[p]++;
    }
    else {
      switch (_buffer[p]) {
        case 'z':
          { _buffer[p] = 'A'; }
          break;
        case 'Z':
          { _buffer[p] = '0'; }
          break;
        case '9':
          {
            _buffer[p] = 'a';
            changeStatus(p + 1);
          }
          break;
        default:
          throw new XMPexception("cannot create a unique name");
      }
    }
  }

  public static String getTempName() throws XMPexception {
    if (_tempInstance == null) _tempInstance = new XMPuniqueName();
    else                       _tempInstance.changeStatus();

    String uniqueString =  new String(_tempInstance._buffer, 0, _tempInstance._pointer + 1);
    return new String("_XCALABLEMP_temp_" + uniqueString);
  }
}
