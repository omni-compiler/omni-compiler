package exc.openmp;

import exc.object.*;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Iterator;

public class OMPtoACCStack {

    private Deque<OMPtoACCStackEntry> stack = new ArrayDeque<>();

    public OMPtoACCStack() {
    }

    public void push(OMPtoACCStackEntry entry) {
        stack.push(entry);
    }

    public OMPtoACCStackEntry pop() {
        return stack.pop();
    }

    public boolean isPop(Xobject xobj) {
        OMPtoACCStackEntry entry = stack.peekFirst();
        if (entry == null) {
            return false;
        }

        if (entry.getXobj() != null) {
            if (entry.getXobj() == xobj) {
                return true;
            }
            return false;
        }

        return true;
    }

    public boolean isInTaskOffload() {
        Iterator<OMPtoACCStackEntry> it = stack.iterator();

        // 'stack' contains self.
        // So, exclude self from the search.
        it.next();

        while (it.hasNext()) {
            OMPtoACCStackEntry e = (OMPtoACCStackEntry) it.next();

            // TODO: target, target teams, target parallel, ...
            switch (e.getPragma()) {
            case TARGET:
            case TARGET_DATA:
            case TARGET_TEAMS:
            case TARGET_PARALLEL:
            case TARGET_PARALLEL_LOOP:
            case TARGET_TEAMS_DISTRIBUTE_PARALLEL_LOOP:
            case TARGET_TEAMS_DISTRIBUTE:
            case DISTRIBUTE_PARALLEL_LOOP:
            case DISTRIBUTE:
                return true;
            }
        }

        return false;
    }

    public boolean isInTaskOffloadWithForLoop() {
        Iterator<OMPtoACCStackEntry> it = stack.iterator();

        // 'stack' contains self.
        // So, exclude self from the search.
        it.next();

        while (it.hasNext()) {
            OMPtoACCStackEntry e = (OMPtoACCStackEntry) it.next();

            // TODO: distribute parallel for, distribute,
            //       for, target teams distribute, target parallel for,
            //       teams distributea, teams distribute parallel for
            switch (e.getPragma()) {
            case TARGET_PARALLEL_LOOP:
            case TARGET_TEAMS_DISTRIBUTE_PARALLEL_LOOP:
            case TARGET_TEAMS_DISTRIBUTE:
            case DISTRIBUTE_PARALLEL_LOOP:
            case DISTRIBUTE:
                return true;
            case TEAMS_DISTRIBUTE:
            case TEAMS_DISTRIBUTE_PARALLEL_LOOP:
            case PARALLEL_FOR:
            case FOR:
                if (isInTaskOffload()) {
                    return true;
                }
                return false;
            }
        }

        return false;
    }
}
