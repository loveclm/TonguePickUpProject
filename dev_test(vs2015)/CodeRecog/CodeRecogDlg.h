
// CodeRecogDlg.h : header file
//

#pragma once

// GENERATED BY MFC TEMPLATE

class CTongueDetectionDlg : public CDialogEx
{
// Construction
public:
	CTongueDetectionDlg(CWnd* pParent = NULL);	// standard constructor

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_CODERECOG_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support

protected:
	HICON m_hIcon;
	
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()

public:
	afx_msg void OnDetectTongueUpdatedAlgorithm();
	BOOL m_bAllFiles;
};
