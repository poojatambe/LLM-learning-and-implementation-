from fastapi import FastAPI, File, UploadFile, Body, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from document_QA import pdf_QA
# from pydantic import BaseModel
# from enum import Enum
import os


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# class DropDown(str, Enum):
#     option1 = 'cohere_generate'
#     option2 = 'cohere_chat'


@app.post('/PDF_QA')
async def upload_pdf(key: str = Body(...), 
                     que: str = Body(...), 
                    #  llm_name: DropDown = Form(...), 
                     file: UploadFile = File(...)):
    if not os.path.exists('./folder'):
        os.makedirs('./folder')
    with open(os.path.join('./folder', file.filename), 'wb') as f:
        pdf_content = await file.read()
        f.write(pdf_content)
    pdf_path = os.path.join('./folder', file.filename)
    results = pdf_QA(key, pdf_path, que)
    return JSONResponse(
        content=results
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)