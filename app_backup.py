"""
Build txtai workflows.

Based on this example: https://github.com/neuml/txtai/blob/master/examples/workflows.py
"""

import os

import nltk
import yaml

import pandas as pd
import streamlit as st

from txtai.embeddings import Documents, Embeddings
from txtai.pipeline import Segmentation, Summary, Tabular, Textractor, Translation
from txtai.workflow import ServiceTask, Task, UrlTask, Workflow


class Process:
    """
    Container for an active Workflow process instance.
    """

    @staticmethod
    @st.cache_resource(ttl=60 * 60, max_entries=3, show_spinner=False)
    def get(components, data):
        """
        Lookup or creates a new workflow process instance.

        Args:
            components: input components
            data: initial data, only passed when indexing

        Returns:
            Process
        """

        process = Process(data)

        # Build workflow
        with st.spinner("Building workflow...."):
            process.build(components)

        return process

    def __init__(self, data):
        """
        Creates a new Process.

        Args:
            data: initial data, only passed when indexing
        """

        # Component options
        self.components = {}

        # Defined pipelines
        self.pipelines = {}

        # Current workflow
        self.workflow = []

        # Embeddings index params
        self.embeddings = None
        self.documents = None
        self.data = data

    def build(self, components):
        """
        Builds a workflow using components.

        Args:
            components: list of components to add to workflow
        """

        # pylint: disable=W0108
        tasks = []
        for component in components:
            component = dict(component)
            wtype = component.pop("type")
            self.components[wtype] = component

            if wtype == "embeddings":
                self.embeddings = Embeddings({**component})
                self.documents = Documents()
                tasks.append(Task(self.documents.add, unpack=False))

            elif wtype == "segmentation":
                self.pipelines[wtype] = Segmentation(**self.components[wtype])
                tasks.append(Task(self.pipelines[wtype]))

        self.workflow = Workflow(tasks)

    def run(self, data):
        """
        Runs a workflow using data as input.

        Args:
            data: input data
        """

        if data and self.workflow:
            # Build tuples for embedding index
            if self.documents:
                data = [(x, element, None) for x, element in enumerate(data)]

            # Process workflow
            for result in self.workflow(data):
                if not self.documents:
                    st.write(result)

            # Build embeddings index
            if self.documents:
                # Cache data
                self.data = list(self.documents)

                with st.spinner("Building embedding index...."):
                    self.embeddings.index(self.documents)
                    self.documents.close()

                # Clear workflow
                self.documents, self.pipelines, self.workflow = None, None, None

    def search(self, query):
        """
        Runs a search.

        Args:
            query: input query
        """

        if self.embeddings and query:
            st.markdown(
                """
            <style>
            table td:nth-child(1) {
                display: none
            }
            table th:nth-child(1) {
                display: none
            }
            table {text-align: left !important}
            </style>
            """,
                unsafe_allow_html=True,
            )

            limit = min(5, len(self.data))

            results = []
            for result in self.embeddings.search(query, limit):
                # Tuples are returned when an index doesn't have stored content
                if isinstance(result, tuple):
                    uid, score = result
                    results.append({"text": self.find(uid), "score": f"{score:.2}"})
                else:
                    if "id" in result and "text" in result:
                        result["text"] = self.content(result.pop("id"), result["text"])
                    if "score" in result and result["score"]:
                        result["score"] = f'{result["score"]:.2}'

                    results.append(result)

            df = pd.DataFrame(results)
            st.write(df.to_html(escape=False), unsafe_allow_html=True)

    def find(self, key):
        """
        Lookup record from cached data by uid key.

        Args:
            key: id to search for

        Returns:
            text for matching id
        """

        # Lookup text by id
        if self.data:
            print(self.data)
            # text = [text for uid, text, _ in self.data if uid == key][0]
            for uid, text, _ in self.data:
                if uid == key:
                    return text[0]
        else:
            text = None
        return self.content(key, text)

    def content(self, uid, text):
        """
        Builds a content reference for uid and text.

        Args:
            uid: record id
            text: record text

        Returns:
            content
        """

        if uid and uid.lower().startswith("http"):
            return f"<a href='{uid}' rel='noopener noreferrer' target='blank'>{text}</a>"

        return text


class Application:
    """
    Main application.
    """

    def __init__(self, directory):
        """
        Creates a new application.
        """

        # Workflow configuration directory
        self.directory = directory

    def default(self, names):
        """
        Gets default workflow index.

        Args:
            names: list of workflow names

        Returns:
           default workflow index
        """

        # Get names as lowercase to match case-insensitive
        lnames = [name.lower() for name in names]

        # Get default workflow param
        params = st.experimental_get_query_params()
        index = params.get("default")
        index = index[0].lower() if index else 0

        # Lookup index of workflow name, add 1 to account for "--"
        if index and index in lnames:
            return lnames.index(index) + 1

        # Workflow not found, default to index 0
        return 0

    def load(self, components):
        """
        Load an existing workflow file.

        Args:
            components: list of components to load

        Returns:
            (names of components loaded, workflow config)
        """

        with open(os.path.join(self.directory, "config.yml"), encoding="utf-8") as f:
            config = yaml.safe_load(f)

        names = [row["name"] for row in config]
        files = [row["file"] for row in config]

        selected = st.selectbox("Load workflow", ["--"] + names, self.default(names))
        if selected != "--":
            index = [x for x, name in enumerate(names) if name == selected][0]
            with open(os.path.join(self.directory, files[index]), encoding="utf-8") as f:
                workflow = yaml.safe_load(f)

            st.markdown("---")

            # Get tasks for first workflow
            tasks = list(workflow["workflow"].values())[0]["tasks"]
            selected = []

            for task in tasks:
                name = task.get("action", task.get("task"))
                if name in components:
                    selected.append(name)
                elif name in ["index", "upsert"]:
                    selected.append("embeddings")

            return (selected, workflow)

        return (None, None)

    def state(self, key):
        """
        Lookup a session state variable.

        Args:
            key: variable key

        Returns:
            variable value
        """

        if key in st.session_state:
            return st.session_state[key]

        return None

    def appsetting(self, workflow, name):
        """
        Looks up an application configuration setting.

        Args:
            workflow: workflow configuration
            name: setting name

        Returns:
            app setting value
        """

        if workflow:
            config = workflow.get("app")
            if config:
                return config.get(name)

        return None

    def setting(self, config, name, default=None):
        """
        Looks up a component configuration setting.

        Args:
            config: component configuration
            name: setting name
            default: default setting value

        Returns:
            setting value
        """

        return config.get(name, default) if config else default

    def text(self, label, component, config, name, default=None):
        """
        Create a new text input field.

        Args:
            label: field label
            component: component name
            config: component configuration
            name: setting name
            default: default setting value

        Returns:
            text input field value
        """

        default = self.setting(config, name, default)
        if not default:
            default = ""
        elif isinstance(default, list):
            default = ",".join(default)
        elif isinstance(default, dict):
            default = ",".join(default.keys())

        st.caption(label)
        st.code(default, language="yaml")
        return default

    def number(self, label, component, config, name, default=None):
        """
        Creates a new numeric input field.

        Args:
            label: field label
            component: component name
            config: component configuration
            name: setting name
            default: default setting value

        Returns:
            numeric value
        """

        value = self.text(label, component, config, name, default)
        return int(value) if value else None

    def boolean(self, label, component, config, name, default=False):
        """
        Creates a new checkbox field.

        Args:
            label: field label
            component: component name
            config: component configuration
            name: setting name
            default: default setting value

        Returns:
            boolean value
        """

        default = self.setting(config, name, default)

        st.caption(label)
        st.markdown(":white_check_mark:" if default else ":white_large_square:")
        return default

    def select(self, label, component, config, name, options, default=0):
        """
        Creates a new select box field.

        Args:
            label: field label
            component: component name
            config: component configuration
            name: setting name
            options: list of dropdown options
            default: default setting value

        Returns:
            boolean value
        """

        index = self.setting(config, name)
        index = [x for x, option in enumerate(options) if option == default]

        # Derive default index
        default = index[0] if index else default

        st.caption(label)
        st.code(options[default], language="yaml")
        return options[default]

    def split(self, text):
        """
        Splits text on commas and returns a list.

        Args:
            text: input text

        Returns:
            list
        """

        return [x.strip() for x in text.split(",")]

    def options(self, component, workflow, index):
        """
        Extracts component settings into a component configuration dict.

        Args:
            component: component type
            workflow: existing workflow, can be None
            index: task index

        Returns:
            dict with component settings
        """

        # pylint: disable=R0912, R0915
        options = {"type": component}

        # Lookup component configuration
        #   - Runtime components have config defined within tasks
        #   - Pipeline components have config defined at workflow root
        config = None
        if workflow:
            config = workflow.get(component)

        if component == "embeddings":
            st.markdown(f"** {index + 1}.) Embeddings Index**  \n*Index workflow output*")
            options["path"] = self.text("Embeddings model path", component, config, "path", "sentence-transformers/nli-mpnet-base-v2")
            options["upsert"] = self.boolean("Upsert", component, config, "upsert")
            options["content"] = self.boolean("Content", component, config, "content")

        elif component in ("segmentation"):
            st.markdown(f"** {index + 1}.) Segment**  \n*Split text into semantic units*")

            options["sentences"] = self.boolean("Split sentences", component, config, "sentences")
            options["lines"] = self.boolean("Split lines", component, config, "lines")
            options["paragraphs"] = self.boolean("Split paragraphs", component, config, "paragraphs")
            options["join"] = self.boolean("Join tokenized", component, config, "join")
            options["minlength"] = self.number("Min section length", component, config, "minlength")


        st.markdown("---")

        return options

    def yaml(self, components):
        """
        Builds a yaml string for components.

        Args:
            components: list of components to export to YAML

        Returns:
            (workflow name, YAML string)
        """

        data = {"app": {"data": self.state("data"), "query": self.state("query")}}
        tasks = []
        name = None

        for component in components:
            component = dict(component)
            name = wtype = component.pop("type")

            if wtype == "embeddings":
                upsert = component.pop("upsert")

                data[wtype] = component
                data["writable"] = True

                name = "index"
                tasks.append({"action": "upsert" if upsert else "index"})

            elif wtype == "segmentation":
                data[wtype] = component
                tasks.append({"action": wtype})


        # Add in workflow
        data["workflow"] = {name: {"tasks": tasks}}

        return (name, yaml.dump(data))

    def data(self, workflow):
        """
        Gets input data.

        Args:
            workflow: workflow configuration

        Returns:
            input data
        """

        # Get default data setting
        data = self.appsetting(workflow, "data")
        if not self.appsetting(workflow, "query"):
            data = st.text_input("Input", value=data)

        # Save data state
        st.session_state["data"] = data

        # Wrap data as list for workflow processing
        return [data]

    def query(self, workflow, index):
        """
        Gets input query.

        Args:
            workflow: workflow configuration
            index: True if this is an indexing workflow

        Returns:
            input query
        """

        default = self.appsetting(workflow, "query")
        default = default if default else ""

        # Get query if this is an indexing workflow
        query = st.text_input("Query", value=default) if index else None

        # Save query state
        st.session_state["query"] = query

        return query

    def process(self, workflow, components, index):
        """
        Processes the current application action.

        Args:
            workflow: workflow configuration
            components: workflow components
            index: True if this is an indexing workflow
        """

        # Get input data and initialize query
        data = self.data(workflow)
        query = self.query(workflow, index)

        # Get workflow process
        process = Process.get(components, data if index else None)

        # Run workflow process
        process.run(data)

        # Run search
        if index:
            process.search(query)

    def run(self):
        """
        Runs Streamlit application.
        """

        with st.sidebar:
            st.image("https://github.com/neuml/txtai/raw/master/logo.png", width=256)
            st.markdown("# Workflow builder  \n*Build and apply workflows to data*  ")
            st.markdown("Workflows combine machine-learning pipelines together to aggregate logic. This application provides a number of pre-configured workflows to get a feel of how they work. Workflows can be exported and run locally through FastAPI. Read more on [GitHub](https://github.com/neuml/txtai) and in the [Docs](https://neuml.github.io/txtai/workflow/).")
            st.markdown("---")

            # Component configuration
            components = ["embeddings", "segmentation"]

            selected, workflow = self.load(components)
            if selected:
                # Get selected options
                components = [self.options(component, workflow, x) for x, component in enumerate(selected)]

        if selected:
            # Process current action
            self.process(workflow, components, "embeddings" in selected)

            with st.sidebar:
                # Generate export button after workflow is complete
                _, config = self.yaml(components)
                st.download_button("Export", config, file_name="workflow.yml", help="Export the API workflow as YAML")
        else:
            st.info("Select a workflow from the sidebar")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # pylint: disable=W0702
    try:
        nltk.sent_tokenize("This is a test. Split")
    except:
        nltk.download("punkt")

    # Create and run application
    app = Application("workflows")
    app.run()
