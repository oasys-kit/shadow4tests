import os, sys

from PyQt5 import QtGui, QtWidgets
from PyQt5.QtGui import QPalette, QColor, QFont
from PyQt5.QtWidgets import QApplication, QFileDialog

from orangewidget import gui
from orangewidget.settings import Setting
from oasys.widgets import gui as oasysgui
from oasys.widgets import congruence

# from orangecontrib.esrf.shadow.util.python_script import PythonConsole

from orangecontrib.shadow.util.shadow_objects import ShadowBeam
from orangecontrib.shadow.util.shadow_util import ShadowCongruence, ShadowPlot
from oasys.widgets import widget
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QTextCursor

import inspect
import numpy
import Shadow

from oasys.util.oasys_util import TriggerIn, TriggerOut, EmittingStream

from orangecontrib.shadow4.util.shadow4_objects import ShadowData
from orangecontrib.shadow4.util.shadow4_util import ShadowCongruence as ShadowCongruence4


"""
Tools to compare beams from shadow3 and Shadow4
"""
import numpy
from srxraylib.plot.gol import plot_scatter
import Shadow
from numpy.testing import assert_almost_equal


def check_six_columns_mean_and_std(beam3, beam4, do_plot=True, do_assert=False, assert_value=1e-2, to_meters=1.0, good_only=True):

    raysnew = beam4.rays
    rays = beam3.rays

    if good_only:
        indices = numpy.where(rays[:,9] > 0 )[0]
        print(indices)
        rays = rays[indices, :].copy()
        raysnew = raysnew[indices, :].copy()

    if do_plot:
        # plot_scatter(rays[:,1],rays[:,0],title="Trajectory shadow3",show=False)
        # plot_scatter(raysnew[:,1],raysnew[:,0],title="Trajectory shadow4")


        plot_scatter(rays[:,3],rays[:,5],title="Divergences shadow3",show=False)
        plot_scatter(raysnew[:,3],raysnew[:,5],title="Divergences shadow4")

        plot_scatter(rays[:,0],rays[:,2],title="Real Space shadow3",show=False)
        plot_scatter(raysnew[:,0],raysnew[:,2],title="Real Space shadow4")

        #
        b3 = Shadow.Beam()
        b3.rays = rays

        b4 = Shadow.Beam()
        b4.rays = raysnew
        Shadow.ShadowTools.histo1(b3,11,ref=23,nolost=1)
        Shadow.ShadowTools.histo1(b4,11,ref=23,nolost=1)



    print("Comparing...")
    for i in range(6):

        m0 = (raysnew[:,i]).mean()
        m1 = (rays[:,i]*to_meters).mean()
        print("\ncol %d, mean sh3, sh4, |sh4-sh3|: %10g  %10g  %10g"%(i+1,m1,m0,numpy.abs(m0-m1)))
        std0 = raysnew[:,i].std()
        std1 = (rays[:,i]*to_meters).std()
        print("col %d, stdv sh3, sh4, |sh4-sh3|: %10g  %10g  %10g"%(i+1,std1,std0,numpy.abs(std0-std1)))

        if do_assert:
            assert(numpy.abs(m0-m1) < assert_value)
            assert(numpy.abs(std0-std1) < assert_value)

def check_almost_equal(beam3, beam4, do_assert=True, display_ray_number=10, level=1, skip_columns=[]):

    print("\ncol#   shadow3  shadow4")
    for i in range(18):

        txt = "col%d   %20.10f  %20.10f  " % (i + 1, beam3.rays[display_ray_number, i], beam4.rays[display_ray_number, i])
        if (i+1) in skip_columns:
            print(txt+"**column not asserted**")
        else:
            print(txt)
            if do_assert:
                if i in [13,14]: # angles
                    assert_almost_equal( numpy.mod(beam3.rays[:, i], numpy.pi), numpy.mod(beam4.rays[:, i], numpy.pi), level)
                else:
                    assert_almost_equal (beam3.rays[:,i], beam4.rays[:,i], level)



class ShadowCompareBeam3Beam4(widget.OWWidget):

    name = "Shadow Python Script"
    description = "Shadow Python Script"
    icon = "icons/python_script.png"
    maintainer = "Manuel Sanchez del Rio"
    maintainer_email = "srio(@at@)esrf.eu"
    priority = 5
    category = "Tools"
    keywords = ["script"]

    inputs = [("Input Beam", ShadowBeam, "setBeam3"),
              ("Shadow Data", ShadowData, "set_shadow_data")]

    input_shadow_data=None

    # sampFactNxNyForProp = Setting(1.0) #0.6 #sampling factor for adjusting nx, ny (effective if > 0)
    # nMacroElec = Setting(500000) #T otal number of Macro-Electrons (Wavefronts)
    # nMacroElecAvgOneProc = Setting(5) # Number of Macro-Electrons (Wavefronts) to average on each node (for MPI calculations)
    # nMacroElecSavePer = Setting(20) # Saving periodicity (in terms of Macro-Electrons) for the Resulting Intensity
    # srCalcMeth = Setting(1) # SR calculation method (1- undulator)
    # srCalcPrec = Setting(0.01) # SR calculation rel. accuracy
    # strIntPropME_OutFileName = Setting("output_srw_script_me.dat")
    # _char = Setting(0)


    script_file_flag = Setting(0)
    script_file_name = Setting("tmp.py")
    source_flag = Setting(0)
    source_file_name = Setting("begin.dat")
    do_run = Setting(1)
    iwrite = Setting(1)

    #
    #
    #
    IMAGE_WIDTH = 890
    IMAGE_HEIGHT = 680

    # want_main_area=1

    is_automatic_run = Setting(True)

    error_id = 0
    warning_id = 0
    info_id = 0

    MAX_WIDTH = 1320
    MAX_HEIGHT = 700

    CONTROL_AREA_WIDTH = 405
    TABS_AREA_HEIGHT = 560

    # srw_live_propagation_mode = "Unknown"




    def __init__(self, show_automatic_box=True, show_general_option_box=True):
        super().__init__() # show_automatic_box=show_automatic_box)


        geom = QApplication.desktop().availableGeometry()
        self.setGeometry(QRect(round(geom.width()*0.05),
                               round(geom.height()*0.05),
                               round(min(geom.width()*0.98, self.MAX_WIDTH)),
                               round(min(geom.height()*0.95, self.MAX_HEIGHT))))

        self.setMaximumHeight(self.geometry().height())
        self.setMaximumWidth(self.geometry().width())

        self.controlArea.setFixedWidth(self.CONTROL_AREA_WIDTH)

        self.general_options_box = gui.widgetBox(self.controlArea, "General Options", addSpace=True, orientation="horizontal")
        self.general_options_box.setVisible(show_general_option_box)

        if show_automatic_box :
            gui.checkBox(self.general_options_box, self, 'is_automatic_run', 'Automatic Execution')


        #
        #
        #
        button_box = oasysgui.widgetBox(self.controlArea, "", addSpace=False, orientation="horizontal")

        button = gui.button(button_box, self, "Compare Beams", callback=self.compare_beams)
        font = QFont(button.font())
        font.setBold(True)
        button.setFont(font)
        palette = QPalette(button.palette()) # make a copy of the palette
        palette.setColor(QPalette.ButtonText, QColor('Dark Blue'))
        button.setPalette(palette) # assign new palette
        button.setFixedHeight(45)

        # button = gui.button(button_box, self, "Reset Fields", callback=self.callResetSettings)
        # font = QFont(button.font())
        # font.setItalic(True)
        # button.setFont(font)
        # palette = QPalette(button.palette()) # make a copy of the palette
        # palette.setColor(QPalette.ButtonText, QColor('Dark Red'))
        # button.setPalette(palette) # assign new palette
        # button.setFixedHeight(45)
        # button.setFixedWidth(150)

        gui.separator(self.controlArea)

        gen_box = oasysgui.widgetBox(self.controlArea, "Script Generation", addSpace=False, orientation="vertical", height=530, width=self.CONTROL_AREA_WIDTH-5)

        gui.comboBox(gen_box, self, "script_file_flag", label="write file with script",
                     items=["No", "Yes"], labelWidth=300,
                     sendSelectedValue=False, orientation="horizontal")

        box1 = gui.widgetBox(gen_box, orientation="horizontal")
        oasysgui.lineEdit(box1, self, "script_file_name", "Script File Name", labelWidth=150, valueType=str,
                          orientation="horizontal")
        self.show_at("self.script_file_flag == 1", box1)


        gui.comboBox(gen_box, self, "source_flag", label="source from",
                     items=["Oasys wire", "shadow3 file"], labelWidth=300,
                     sendSelectedValue=False, orientation="horizontal")

        box1 = gui.widgetBox(gen_box, orientation="horizontal")
        oasysgui.lineEdit(box1, self, "source_file_name", "Source File Name", labelWidth=150, valueType=str,
                          orientation="horizontal")
        self.show_at("self.source_flag == 1", box1)

        gui.comboBox(gen_box, self, "do_run", label="run shadow3 (in script)",
                     items=["No (only definitions)", "Yes (define and run)"], labelWidth=300,
                     sendSelectedValue=False, orientation="horizontal")

        gui.comboBox(gen_box, self, "iwrite", label="when running, write shadow files",
                     items=["No", "Yes"], labelWidth=300,
                     sendSelectedValue=False, orientation="horizontal")


        tabs_setting = oasysgui.tabWidget(self.mainArea)
        tabs_setting.setFixedHeight(self.IMAGE_HEIGHT)
        tabs_setting.setFixedWidth(self.IMAGE_WIDTH)

        # tab_scr = oasysgui.createTabPage(tabs_setting, "Python Script")
        tab_out = oasysgui.createTabPage(tabs_setting, "System Output")

        self.pythonScript = oasysgui.textArea(readOnly=False)
        self.pythonScript.setStyleSheet("background-color: white; font-family: Courier, monospace;")
        self.pythonScript.setMaximumHeight(self.IMAGE_HEIGHT - 250)

        # script_box = oasysgui.widgetBox(tab_scr, "", addSpace=False, orientation="vertical", height=self.IMAGE_HEIGHT - 10, width=self.IMAGE_WIDTH - 10)
        # script_box.layout().addWidget(self.pythonScript)

        # console_box = oasysgui.widgetBox(script_box, "", addSpace=True, orientation="vertical",
        #                                   height=150, width=self.IMAGE_WIDTH - 10)
        #
        # self.console = PythonConsole(self.__dict__, self)
        # console_box.layout().addWidget(self.console)

        self.shadow_output = oasysgui.textArea()

        out_box = oasysgui.widgetBox(tab_out, "System Output", addSpace=True, orientation="horizontal", height=self.IMAGE_WIDTH - 45)
        out_box.layout().addWidget(self.shadow_output)

        #############################

        # button_box = oasysgui.widgetBox(tab_scr, "", addSpace=True, orientation="horizontal")

        # gui.button(button_box, self, "Run Script", callback=self.execute_script, height=40)
        # gui.button(button_box, self, "Save Script to File", callback=self.save_script, height=40)

        gui.rubber(self.controlArea)

        self.process_showers()

    def callResetSettings(self):
        pass

    # def execute_script(self):
    #
    #     self._script = str(self.pythonScript.toPlainText())
    #     self.console.write("\nRunning script:\n")
    #     self.console.push("exec(_script)")
    #     self.console.new_prompt(sys.ps1)
    #
    #
    # def save_script(self):
    #     # file_name = QFileDialog.getSaveFileName(self, "Save File to Disk", os.getcwd(), filter='*.py')[0]
    #     file_name = self.script_file_name
    #     if not file_name is None:
    #         if not file_name.strip() == "":
    #             file = open(file_name, "w")
    #             file.write(str(self.pythonScript.toPlainText()))
    #             file.close()


    def setBeam3(self, beam):
        if ShadowCongruence.checkEmptyBeam(beam):
            if ShadowCongruence.checkGoodBeam(beam):
                # sys.stdout = EmittingStream(textWritten=self.writeStdOut)

                self.input_beam = beam

                if self.is_automatic_run:
                    self.compare_beams()

            else:
                QtWidgets.QMessageBox.critical(self, "Error",
                                           "Data not displayable: No good rays or bad content",
                                           QtWidgets.QMessageBox.Ok)

    def set_shadow_data(self, input_data):
        # self.not_interactive = self._check_not_interactive_conditions(input_data)
        #
        # self._on_receiving_input()

        if ShadowCongruence4.check_empty_data(input_data):
            self.input_data = input_data.duplicate()
            if self.is_automatic_run: self.compare_beams()

    def writeStdOut(self, text):
        cursor = self.shadow_output.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.shadow_output.setTextCursor(cursor)
        self.shadow_output.ensureCursorVisible()

    def compare_beams(self):
        self.shadow_output.setText("")
        sys.stdout = EmittingStream(textWritten=self.writeStdOut)
        print(">>>> comparing shadow3 and shadow4 beams")

        fail = 0
        try:
            beam3 = self.input_beam._beam
        except:
            print(">>> Error retrieving beam3")
            fail = 1
        try:
            beam4 = self.input_data.beam
        except:
            print(">>> Error retrieving beam4")
            fail = 1

        if not fail:
            check_six_columns_mean_and_std(beam3, beam4, do_assert=True, do_plot=False)
            check_almost_equal(beam3, beam4, do_assert=True, level=6)




    #     optical_element_list_start = []
    #     optical_element_list_end = []
    #
    #     self.pythonScript.setText("")
    #
    #     for history_element in self.input_beam.getOEHistory():
    #         if not history_element._shadow_source_start is None:
    #             optical_element_list_start.append(history_element._shadow_source_start.src)
    #         elif not history_element._shadow_oe_start is None:
    #             optical_element_list_start.append(history_element._shadow_oe_start._oe)
    #
    #         if not history_element._shadow_source_end is None:
    #             optical_element_list_end.append(history_element._shadow_source_end.src)
    #         elif not history_element._shadow_oe_end is None:
    #             optical_element_list_end.append(history_element._shadow_oe_end._oe)
    #
    #
    #     try:
    #         if self.script_file_flag == 0:
    #             script_file = ""
    #         else:
    #             script_file = self.script_file_name
    #         self.pythonScript.setText(make_python_script_from_list(optical_element_list_start,
    #                                                                script_file=script_file,
    #                                                                source_flag=self.source_flag,
    #                                                                source_file_name=self.source_file_name,
    #                                                                do_run=self.do_run,
    #                                                                iwrite=self.iwrite) )
    #     except:
    #         self.pythonScript.setText(
    #             "Problem in writing python script:\n" + str(sys.exc_info()[0]) + ": " + str(sys.exc_info()[1]))
    #
    #     if self.script_file_flag:
    #         self.save_script()


        # if not self.input_shadow_data is None:
        #     self.pythonScript.setText("")
        #
        #     try:
        #         received_light_source = self.input_shadow_data.get_srw_beamline().get_light_source()
        #
        #         if not (isinstance(received_light_source, SRWBendingMagnetLightSource) or isinstance(received_light_source, SRWUndulatorLightSource)):
        #             raise ValueError("ME Script is not available with this source")
        #
        #         _char = 0 if self._char == 0 else 4
        #
        #         parameters = [self.sampFactNxNyForProp,
        #                       self.nMacroElec,
        #                       self.nMacroElecAvgOneProc,
        #                       self.nMacroElecSavePer,
        #                       self.srCalcMeth,
        #                       self.srCalcPrec,
        #                       self.strIntPropME_OutFileName,
        #                       _char]
        #
        #         self.pythonScript.setText(self.input_shadow_data.get_srw_beamline().to_python_code([self.input_shadow_data.get_srw_wavefront(), True, parameters]))
        #     except Exception as e:
        #         self.pythonScript.setText("Problem in writing python script:\n" + str(sys.exc_info()[0]) + ": " + str(sys.exc_info()[1]))
        #
        #         if self.IS_DEVELOP: raise e


#
# automatic creation of python scripts
#



if __name__ == "__main__":
    import sys
    import Shadow

    # class MyBeam():
    #     pass
    # beam_to_analize = Shadow.Beam()
    # beam_to_analize.load("/users/srio/Oasys/star.01")
    # my_beam = MyBeam()
    # my_beam._beam = beam_to_analize

    a = QApplication(sys.argv)
    ow = ShadowCompareBeam3Beam4()
    # ow.pythonScript.setText("#\n#\n#\nprint('Hello world')\n")
    ow.show()
    # ow.input_beam = my_beam
    a.exec_()
    ow.saveSettings()