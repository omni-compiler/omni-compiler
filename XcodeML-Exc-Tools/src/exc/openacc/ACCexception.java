/* -*- Mode: java; c-basic-offset:2 ; indent-tabs-mode:nil ; -*- */
package exc.openacc;

public class ACCexception extends Exception {
  private static final long serialVersionUID = 3370547121289650311L;

  public ACCexception() {
    super();
  }

  public ACCexception(String msg) {
    super(msg);
  }
}
