from setuptools import setup

setup(
    name='ilab_tools',
    version='0.0.6',
    packages=['ilab_tools'],
    description='Python library of iLab internal tools.',
    long_description=open('README.md').read(),
    install_requires=open('requirements.txt').read().splitlines(),
    include_package_data=True,

    extras_require={
        "chroma": ["numpy>=2.1.3", "pandas>=2.2.3", "chromadb==0.5.9", "scikit-learn>=1.5.2"],
        "azure": ["pandas>=2.0.3", "pyodbc==5.0.1"],
        "zabbix": ["py-zabbix==1.1.7"],
        "twitter": ["requests==2.32.3"],
        "slack": ["slack_sdk==3.33.3"],
        "mysql": ["sqlalchemy<2.0", "pymysql>=1.0.2"], #PyMySQL==1.1.1 SQLAlchemy==2.0.36
        "mongodb": ["pymongo==4.10.1"],
        "faiss": ["faiss-cpu==1.8.0", "google-cloud-storage>=2.18.2", "pytest>=8.3.3"],
        "df_operations": ["pandas>=2.2.3"],
        "gcp": ["pandas>=2.2.3", "google-cloud-bigquery>=3.27.0", "google-cloud-storage>=2.18.2", "oauth2client>=4.1.3", "python-dotenv>=1.0.1", "gspread>=6.1.4"],
    }
)
