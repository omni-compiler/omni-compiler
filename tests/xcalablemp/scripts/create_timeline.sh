#!/bin/sh

## Set Arguments
OUTPUT_DIR=$1
JOB_NUM=$2
ELAPSE_TIME_SEC=$3
BASE_TESTDIR=$4
shift; shift; shift; shift
TESTDIRS=$@
CURRENT_DIR=`pwd`

## Set static variable
TIMELINE_FILE=timeline.dat
STATIC_COLORS=('red' 'darkred' 'mistyrose' 'steelblue' 'lightsteelblue' 'darkblue' 'yellowgreen' 'darkolivegreen' 'lime' 'magenta' 'darkmagenta' 'mediumpurple' '#cbb69d' '#603913' '#c69c6e')
NUM_OF_STATIC_COLORS=${#STATIC_COLORS[*]}
TIME_LINE_DATA=""
NUM_OF_DIRS=0

## Set Data of master
for dat in autogen.dat configure.dat make.dat; do
    FILE=${CURRENT_DIR}/$dat
    if [ -f ${FILE} ]; then
	JOB_NAME=`head -1 ${FILE}`
	NODE_NAME=`head -2 ${FILE} | tail -1`
	START_TIME=`head -3 ${FILE} | tail -1`
	END_TIME=`tail -1 ${FILE}`
	TIME_LINE_DATA+="['"${NODE_NAME}"',\t"
	TIME_LINE_DATA+="'"${JOB_NAME}"',\t"
	TIME_LINE_DATA+=" new Date(${START_TIME}),\t"
	TIME_LINE_DATA+=" new Date(${END_TIME})],\n"
	NUM_OF_DIRS=`expr $NUM_OF_DIRS + 1`
    fi
done

## Set Data of slave
for subdir in ${TESTDIRS}; do
    FILE=${BASE_TESTDIR}/$subdir/${TIMELINE_FILE}
    if [ -f ${FILE} ]; then
	JOB_NAME=`head -1 ${FILE}`
	NODE_NAME=`head -2 ${FILE} | tail -1`
	START_TIME=`head -3 ${FILE} | tail -1`
	END_TIME=`tail -1 ${FILE}`
	TIME_LINE_DATA+="['"${NODE_NAME}"',\t"
	TIME_LINE_DATA+="'"${JOB_NAME}"',\t"
	TIME_LINE_DATA+=" new Date(${START_TIME}),\t"
	TIME_LINE_DATA+=" new Date(${END_TIME})],\n"
	NUM_OF_DIRS=`expr $NUM_OF_DIRS + 1`
    fi
done
TIME_LINE_DATA=`echo -e $TIME_LINE_DATA | sort -n`

## Create Directory
mkdir -p $OUTPUT_DIR
OUTPUT_FILE=${OUTPUT_DIR}"/"${JOB_NUM}.html

## Set bar colors
COLORS=""
NUM=0
for subdir in `seq 1 $NUM_OF_DIRS`; do
    REST_NUM=`expr $NUM % $NUM_OF_STATIC_COLORS`
    COLORS+="'"${STATIC_COLORS[${REST_NUM}]}"',"
    NUM=`expr $NUM + 1`
done

## OUTPUT
OUTPUT=`cat <<EOF
<script type="text/javascript" src="https://www.google.com/jsapi?autoload={'modules':[{'name':'visualization', \n
                                    'version':'1','packages':['timeline']}]}"></script> \n
<script type="text/javascript"> \n
\n
google.setOnLoadCallback(drawChart);\n
function drawChart() \n
{\n
    var container = document.getElementById('timeline');\n
    var chart = new google.visualization.Timeline(container);\n
\n
    var dataTable = new google.visualization.DataTable();\n
    dataTable.addColumn({ type: 'string', id: 'Position' });\n
    dataTable.addColumn({ type: 'string', id: 'JobName' });\n
    dataTable.addColumn({ type: 'date', id: 'Start' });\n
    dataTable.addColumn({ type: 'date', id: 'End' });\n
    dataTable.addRows([\n
    ${TIME_LINE_DATA}
    ]);\n
\n
    var options = {\n
    hAxis:{ format:'kk:mm:ss'},\n
    colors: [\n
    ${COLORS}\n
    ]};\n
\n
    chart.draw(dataTable, options);\n
  }\n
</script>\n
\n
<div id="timeline" style="width: 1500px; height: 250px;"></div>
<p>
Elapse time is $ELAPSE_TIME_SEC sec.
</p>
EOF`

echo -e ${OUTPUT} > ${OUTPUT_FILE}
